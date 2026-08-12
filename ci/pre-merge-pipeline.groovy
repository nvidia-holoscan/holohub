#!/usr/bin/env groovy
// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Trusted Jenkins entry point for GitLab merge-request validation.
 *
 * Configure the Jenkins job to load this script and its helpers from protected
 * `main`. The stages below explicitly create and test a source-into-target
 * merge; no pipeline code is loaded from the untrusted merge-request ref.
 */

def Utils
def Stage
node {
    deleteDir()
    checkout scm
    Utils = load 'ci/utils.groovy'
    Stage = load 'ci/stage.groovy'
    Stage.init_utils(Utils)
}

Utils.validate_build_context()
Utils.validate_comment_trigger_helpers()

def mergeRequestBuild = Utils.is_merge_request_build()
def nightlyBuild = Utils.is_nightly_build()
def manualBuild = Utils.is_manual_build()
def pushRequested = Utils.push_requested()
def submitToCdash = nightlyBuild || manualBuild || pushRequested
def cdashSubmissionReason = nightlyBuild
    ? 'nightly cron'
    : (manualBuild ? 'manual Build Now' : '[push] comment')

echo(
    submitToCdash
        ? "CDash submission enabled (${cdashSubmissionReason})."
        : 'CDash submission disabled; build and test output remains in Jenkins.'
)

if (Utils.should_skip_draft_merge_request()) {
    echo 'Skipping automatic validation for a draft merge request.'
    updateGitlabCommitStatus(name: 'pre-merge', state: 'success')
    currentBuild.description = "MR ${env.gitlabMergeRequestIid}: draft skipped"
    return
}

if (mergeRequestBuild) {
    currentBuild.description =
        "MR ${env.gitlabMergeRequestIid}: ${env.gitlabSourceBranch} -> ${env.gitlabTargetBranch}"
} else {
    currentBuild.description = nightlyBuild
        ? 'Nightly coverage: latest main'
        : 'Manual coverage: latest main'
}

def lintSettings = [
    name: 'lint',
    container_name: 'lint-dind',
    kubernetes_arch: 'amd64',
    cpus: 8,
    memory: '24Gi',
    ephemeral_storage: '50Gi',
    gpus: 0,
    timeout_amount: 90,
    timeout_unit: 'MINUTES',
]

def buildAndTestSettings = [
    [
        name: 'x86_64-cuda13',
        container_name: 'x86-tester-dind',
        kubernetes_arch: 'amd64',
        host_architecture: 'x86_64',
        sdk_architecture: 'x86_64',
        cpus: 13,
        memory: '60Gi',
        ephemeral_storage: '200Gi',
        gpus: 1,
        timeout_amount: 6,
        timeout_unit: 'HOURS',
    ],
    [
        name: 'sbsa-cuda13',
        container_name: 'sbsa-tester-dind',
        kubernetes_arch: 'arm64',
        host_architecture: 'aarch64',
        sdk_architecture: 'aarch64',
        cpus: 13,
        memory: '60Gi',
        ephemeral_storage: '200Gi',
        gpus: 1,
        timeout_amount: 6,
        timeout_unit: 'HOURS',
    ],
]

def flows = [:]

flows[lintSettings.name] = Utils.setup_flow(lintSettings) {
    Stage.code_checkout()
    Stage.check_lint()
}

buildAndTestSettings.each { flowSettings ->
    def settings = flowSettings
    flows[settings.name] = Utils.setup_flow(settings) {
        try {
            Stage.code_checkout()
            Stage.wait_for_docker_daemon()
            Stage.verify_runner(settings.host_architecture)
            def sdkRevision = Stage.checkout_pinned_sdk()

            withEnv([
                "ARCH=${settings.sdk_architecture}",
                'GPU=dgpu',
                'CUDA_MAJOR=13',
            ]) {
                Stage.build_sdk()
                def sdkInstall = Stage.resolve_sdk_install(
                    sdkRevision,
                    settings.host_architecture,
                )
                withEnv(["HOLOSCAN_SDK_INSTALL_DIR=${sdkInstall}"]) {
                    Stage.build_sample(settings.name, submitToCdash, nightlyBuild)
                    Stage.test_sample(settings.name, submitToCdash, nightlyBuild)
                }
            }
        } finally {
            Stage.collect_artifacts(settings.name)
        }
    }
}

if (mergeRequestBuild) {
    def statusNames = flows.keySet().toList() + ['pre-merge']
    gitlabBuilds(builds: statusNames) {
        updateGitlabCommitStatus(name: 'pre-merge', state: 'running')
        try {
            parallel(flows)
            updateGitlabCommitStatus(name: 'pre-merge', state: 'success')
        } catch (Throwable error) {
            updateGitlabCommitStatus(name: 'pre-merge', state: 'failed')
            throw error
        }
    }
} else {
    echo 'No GitLab merge-request context; running full coverage on latest main.'
    parallel(flows)
}
