#!/usr/bin/env groovy
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import groovy.transform.Field

// Shared infrastructure and flow helpers for repository Jenkins pipelines.
@Field private final String BLOSSOM_CLOUD = 'sc-ipp-blossom-prod'
@Field private final String BLOSSOM_NAMESPACE = 'clara'
@Field private final String DIND_IMAGE =
    'gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/ci-dind:26.02.20'
@Field private final String JNLP_IMAGE =
    'jenkins/inbound-agent:3345.v03dee9b_f88fc-1-jdk21'
@Field private final String SDK_REPOSITORY =
    'https://gitlab-master.nvidia.com/holoscan/holoscan-sdk.git'
@Field private final String SDK_REGISTRY = 'gitlab-master.nvidia.com:5005'
@Field private final String SDK_REGISTRY_REPOSITORY =
    "${SDK_REGISTRY}/holoscan/holoscan-sdk"
@Field private final String CAMERA_CREDENTIAL =
    'HOLOSCAN_CAMERA_GITLAB_READ_TOKEN'
@Field private final String SDK_CREDENTIAL = 'HOLOSCAN_SDK_GITLAB_READ_TOKEN'
@Field private final String INTERNAL_GITLAB_PREFIX =
    'https://gitlab-master.nvidia.com/'
// Scheduling policy is intentionally kept near the top of this file so that
// changes are easy to review. NODE_BLACKLIST_ENV is managed by Jenkins;
// FLOW_EXCLUDED_NODES is for repository-local exclusions.
@Field private final String NODE_BLACKLIST_ENV = 'BLACKLISTED_NODES'
@Field private final List<String> FLOW_EXCLUDED_NODES = [
    '2u1g-b650-1788.ipp3a2.colossus',
]

// Keep this list aligned with get_default_driver_versions() in the pinned
// holoscan-5x ci/scripts/utils.groovy. Kubernetes affinity cannot perform a
// numeric comparison on dotted driver versions. Stage.verify_runner() also
// enforces the minimum driver family at runtime.
@Field private final List<String> SUPPORTED_DRIVERS = [
    '580.23', '580.35', '580.40', '580.54', '580.55', '580.65.03',
    '580.65.06', '580.66', '580.76', '580.76.07', '580.82.06',
    '580.82.07', '580.95.05', '580.35-open', '580.40-open',
    '580.86-open', '595.25', '595.25-open', '595.39', '595.45.04',
    '595.45.04-open', '610.15', '610.15-open', '610.36', '610.36-open',
    '610.42', '610.42-open',
]

def get_sdk_repository() {
    return SDK_REPOSITORY
}

def get_sdk_credential() {
    return SDK_CREDENTIAL
}

def get_sdk_registry() {
    return SDK_REGISTRY
}

def get_sdk_build_cache_image(String architecture, String sdkRevision) {
    if (!(architecture in ['x86_64', 'aarch64'])) {
        throw new IllegalArgumentException("Unsupported SDK architecture: ${architecture}")
    }
    if (!(sdkRevision ==~ /[0-9a-f]{40}/)) {
        throw new IllegalArgumentException('SDK revision must be a full lowercase SHA')
    }
    return "${SDK_REGISTRY_REPOSITORY}/build-${architecture}:${sdkRevision.take(9)}"
}

def is_merge_request_build() {
    return env.gitlabMergeRequestIid?.trim() as Boolean
}

def is_nightly_build() {
    return !currentBuild.getBuildCauses(
        'hudson.triggers.TimerTrigger$TimerTriggerCause'
    ).isEmpty()
}

// Jenkins may escape square brackets when it exposes the triggering comment.
def push_requested_comment(String comment) {
    return comment?.contains('[push]') || comment?.contains('\\[push\\]')
}

def push_requested() {
    return push_requested_comment(env.gitlabTriggerPhrase)
}

def validate_build_context() {
    // A manual Jenkins "Build Now" run has no GitLab merge-request metadata.
    // In that case the job's trusted SCM configuration supplies latest main.
    if (!is_merge_request_build()) {
        return
    }

    def required = [
        'gitlabSourceBranch',
        'gitlabTargetBranch',
        'gitlabSourceRepoHttpUrl',
        'gitlabTargetRepoHttpUrl',
    ]
    def missing = required.findAll { !env."${it}"?.trim() }
    if (missing) {
        error("Missing GitLab merge-request environment values: ${missing.join(', ')}")
    }

    def repositoryUrls = [
        env.gitlabSourceRepoHttpUrl,
        env.gitlabTargetRepoHttpUrl,
    ]
    if (repositoryUrls.any { !it.startsWith(INTERNAL_GITLAB_PREFIX) }) {
        error('Pre-merge builds are restricted to internal GitLab repositories')
    }
}

def get_merge_scm() {
    if (!is_merge_request_build()) {
        error('Merge SCM was requested outside a GitLab merge-request build')
    }
    validate_build_context()
    return [
        $class: 'GitSCM',
        branches: [[name: "source/${env.gitlabSourceBranch}"]],
        extensions: [
            [$class: 'CleanCheckout'],
            [$class: 'PruneStaleBranch'],
            [
                $class: 'UserIdentity',
                name: 'Jenkins Merge Bot',
                email: 'jenkins@localhost.com',
            ],
            [
                $class: 'PreBuildMerge',
                options: [
                    fastForwardMode: 'NO_FF',
                    mergeRemote: 'target',
                    mergeTarget: env.gitlabTargetBranch,
                ],
            ],
        ],
        userRemoteConfigs: [
            [
                credentialsId: CAMERA_CREDENTIAL,
                name: 'target',
                url: env.gitlabTargetRepoHttpUrl,
            ],
            [
                credentialsId: CAMERA_CREDENTIAL,
                name: 'source',
                url: env.gitlabSourceRepoHttpUrl,
            ],
        ],
    ]
}

def get_pod_yaml(Map settings) {
    def arch = settings.kubernetes_arch
    def containerName = settings.container_name
    def cpus = settings.cpus
    def memory = settings.memory
    def ephemeralStorage = settings.ephemeral_storage
    def gpuCount = settings.gpus
    def gpuResources = gpuCount
        ? """
          nvidia.com/gpu: "${gpuCount}"
"""
        : ''
    def driverAffinity = gpuCount
        ? """
              - key: nvidia.com/driver_version
                operator: In
                values:
${SUPPORTED_DRIVERS.collect { "                  - \"${it}\"" }.join('\n')}
"""
        : ''
    def excludedNodes = get_excluded_nodes(settings)
    def nodeAffinity = excludedNodes
        ? """
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
${excludedNodes.collect { "                  - \"${it}\"" }.join('\n')}
"""
        : ''

    return """
apiVersion: v1
kind: Pod
spec:
  restartPolicy: Never
  containers:
    - name: jnlp
      image: ${JNLP_IMAGE}
      command:
        - /bin/sh
        - -c
      args:
        - |
          attempt=1
          max_attempts=12
          while [ "\$attempt" -le "\$max_attempts" ]; do
            /usr/local/bin/jenkins-agent && exit 0
            status=\$?
            if [ "\$attempt" -eq "\$max_attempts" ]; then
              echo "JNLP agent failed after \$attempt attempts (exit \$status)" >&2
              exit "\$status"
            fi
            delay=\$((attempt * 10))
            if [ "\$delay" -gt 60 ]; then
              delay=60
            fi
            echo "JNLP agent exited \$status; retrying in \$delay seconds (attempt \$attempt/\$max_attempts)" >&2
            sleep "\$delay"
            attempt=\$((attempt + 1))
          done
          exit 1
    - name: ${containerName}
      image: ${DIND_IMAGE}
      imagePullPolicy: Always
      tty: true
      securityContext:
        privileged: true
      env:
        - name: DOCKER_TLS_CERTDIR
          value: ""
        - name: K8S_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
      resources:
        requests:
          cpu: "${cpus}"
          memory: "${memory}"
          ephemeral-storage: "${ephemeralStorage}"
${gpuResources}
        limits:
          cpu: "${cpus}"
          memory: "${memory}"
          ephemeral-storage: "${ephemeralStorage}"
${gpuResources}
  nodeSelector:
    kubernetes.io/os: linux
    kubernetes.io/arch: ${arch}
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: kubernetes.io/arch
                operator: In
                values:
                  - ${arch}
${nodeAffinity}
${driverAffinity}
  imagePullSecrets:
    - name: clara-holoscan-sdk-read-registry
"""
}

def get_excluded_nodes(Map settings = [:]) {
    def configured = []
    // Use direct Jenkins environment-property access. Dynamic env[...] access
    // is rejected by the Pipeline Groovy sandbox before pod provisioning.
    def globalNodes = env.BLACKLISTED_NODES
    if (globalNodes?.trim()) {
        configured.addAll(globalNodes.split(',') as List)
    }
    configured.addAll(FLOW_EXCLUDED_NODES)
    if (settings.exclude_nodes instanceof Collection) {
        configured.addAll(settings.exclude_nodes)
    } else if (settings.exclude_nodes) {
        configured.addAll(settings.exclude_nodes.toString().split(',') as List)
    }
    return configured.collect { it.toString().trim() }.findAll { it }.unique()
}

def setup_flow(Map settings, Closure body) {
    def flowName = settings.name
    def containerName = settings.container_name
    def advisoryMr = settings.advisory_mr ?: false
    def reportGitlabStatus = settings.report_gitlab_status != false
    def timeoutAmount = settings.timeout_amount
    def timeoutUnit = settings.timeout_unit
    def yaml = get_pod_yaml(settings)

    return {
        podTemplate(
            cloud: BLOSSOM_CLOUD,
            namespace: BLOSSOM_NAMESPACE,
            yaml: yaml,
        ) {
            timeout(time: timeoutAmount, unit: timeoutUnit) {
                node(POD_LABEL) {
                    container(containerName) {
                        def runBody = {
                            if (is_merge_request_build() && advisoryMr) {
                                catchError(
                                    buildResult: 'SUCCESS',
                                    stageResult: 'FAILURE',
                                    catchInterruptions: false,
                                ) {
                                    body()
                                }
                            } else {
                                body()
                            }
                        }
                        if (is_merge_request_build() && reportGitlabStatus) {
                            gitlabCommitStatus(name: flowName) {
                                runBody()
                            }
                        } else {
                            runBody()
                        }
                    }
                }
            }
        }
    }
}

def is_draft_merge_request() {
    return env.gitlabMergeRequestTitle?.startsWith('Draft: ')
}

def is_manual_rebuild_comment(String triggerPhrase) {
    def comment = triggerPhrase?.trim()
    return comment ? (comment ==~ /(?is)^rebuild(?:\s.*)?$/) : false
}

def is_manual_rebuild() {
    return is_manual_rebuild_comment(env.gitlabTriggerPhrase)
}

def should_skip_draft_merge_request() {
    return is_merge_request_build() &&
        is_draft_merge_request() &&
        !is_manual_rebuild()
}

def validate_comment_trigger_helpers() {
    def require = { boolean condition, String message ->
        if (!condition) {
            error("Comment trigger helper self-check failed: ${message}")
        }
    }

    [
        'rebuild',
        'Rebuild',
        'REBUILD',
        'rebuild [push]',
        'Rebuild [push]',
        'rebuild \\[push\\]',
        'rebuild because runner was unavailable',
        "rebuild\n[push]",
    ].each { comment ->
        require(
            is_manual_rebuild_comment(comment),
            "expected manual rebuild for '${comment}'",
        )
    }

    [
        null,
        '',
        'rerun',
        'please rebuild',
    ].each { comment ->
        require(
            !is_manual_rebuild_comment(comment),
            "expected non-manual rebuild for '${comment}'",
        )
    }

    [
        'rebuild [push]',
        'rebuild \\[push\\]',
        'please [push] this',
    ].each { comment ->
        require(push_requested_comment(comment), "expected push for '${comment}'")
    }

    [
        null,
        '',
        'rebuild',
        'rebuild [Push]',
    ].each { comment ->
        require(!push_requested_comment(comment), "expected no push for '${comment}'")
    }
}

return this
