#!/usr/bin/env groovy
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import groovy.transform.Field

// Reusable user-visible stages for repository Jenkins pipelines.
@Field private def Utils = null

def init_utils(utils) {
    Utils = utils
}

// Jenkins loads this trusted helper from main while validating a merge request.
// Prefer the renamed wrapper when the checked-out source contains it, but retain
// the legacy name until the rename change itself has landed on main.
def module_cli_wrapper() {
    return fileExists('holoscan_camera') ? './holoscan_camera' : './holohub'
}

def code_checkout() {
    def mergeRequestBuild = Utils.is_merge_request_build()
    def stageName = mergeRequestBuild ? 'Checkout merged MR' : 'Checkout main'
    stage(stageName) {
        if (mergeRequestBuild) {
            checkout(changelog: true, poll: false, scm: Utils.get_merge_scm())
        } else {
            checkout(changelog: true, poll: false, scm: scm)
        }
        sh '''
            set -eu
            git config --global --add safe.directory "$WORKSPACE"
            echo "Checked-out commit: $(git rev-parse HEAD)"
            echo "Checked-out tree:   $(git rev-parse HEAD^{tree})"
            git status --short
        '''
    }
}

def wait_for_docker_daemon() {
    stage('Wait for Docker') {
        timeout(time: 3, unit: 'MINUTES') {
            sh '''
                set -eu
                until docker info >/dev/null 2>&1; do
                    sleep 2
                done
                docker version
            '''
        }
    }
}

def verify_runner(String expectedArchitecture) {
    stage('Verify runner') {
        withEnv(["EXPECTED_ARCHITECTURE=${expectedArchitecture}"]) {
            sh '''
                set -eu
                actual_arch="$(uname -m)"
                if [ "$actual_arch" != "$EXPECTED_ARCHITECTURE" ]; then
                    echo "Expected $EXPECTED_ARCHITECTURE, found $actual_arch" >&2
                    exit 1
                fi

                driver_version="$(
                    nvidia-smi --query-gpu=driver_version --format=csv,noheader |
                        head -n1 | tr -d ' '
                )"
                driver_major="${driver_version%%.*}"
                case "$driver_major" in
                    ''|*[!0-9]*)
                        echo "Could not parse NVIDIA driver version: $driver_version" >&2
                        exit 1
                        ;;
                esac
                if [ "$driver_major" -lt 580 ]; then
                    echo "NVIDIA R580 or newer is required; found $driver_version" >&2
                    exit 1
                fi

                echo "Node: ${K8S_NODE_NAME:-unknown}"
                echo "Architecture: $actual_arch"
                echo "NVIDIA driver: $driver_version"
                nvidia-smi
            '''
        }
    }
}

def check_lint() {
    stage('Holoscan CLI lint') {
        withEnv(["HOLOSCAN_CAMERA_WRAPPER=${module_cli_wrapper()}"]) {
            sh '''
            set -eu
            "$HOLOSCAN_CAMERA_WRAPPER" list
            "$HOLOSCAN_CAMERA_WRAPPER" modes holoscan_camera_v4l2 --language cpp
            "$HOLOSCAN_CAMERA_WRAPPER" lint
            python3 -m unittest ci.test_report_cdash_test_failure
            git diff --exit-code
        '''
        }
    }
}

def checkout_pinned_sdk() {
    def sdkRevision = readFile('ci/holoscan-sdk.version').trim()
    if (!(sdkRevision ==~ /[0-9a-f]{40}/)) {
        error('ci/holoscan-sdk.version must contain exactly one full lowercase SHA')
    }

    stage('Checkout pinned Holoscan SDK') {
        dir('holoscan-sdk') {
            deleteDir()
            checkout([
                $class: 'GitSCM',
                branches: [[name: sdkRevision]],
                extensions: [
                    [
                        $class: 'CloneOption',
                        depth: 0,
                        honorRefspec: false,
                        noTags: true,
                        shallow: false,
                        timeout: 30,
                    ],
                ],
                userRemoteConfigs: [[
                    credentialsId: Utils.get_sdk_credential(),
                    url: Utils.get_sdk_repository(),
                ]],
            ])
            withEnv([
                "EXPECTED_SDK_REVISION=${sdkRevision}",
                "SDK_CHECKOUT_DIR=${pwd()}",
            ]) {
                sh '''
                    set -eu
                    git config --global --add safe.directory "$SDK_CHECKOUT_DIR"
                    actual_revision="$(
                        git -c safe.directory="$SDK_CHECKOUT_DIR" rev-parse HEAD
                    )"
                    if [ "$actual_revision" != "$EXPECTED_SDK_REVISION" ]; then
                        echo "Expected SDK $EXPECTED_SDK_REVISION, found $actual_revision" >&2
                        exit 1
                    fi
                    echo "Holoscan SDK revision: $actual_revision"
                '''
            }
        }
    }
    return sdkRevision
}

def checkout_sdk_branch(String sdkBranch) {
    if (!(sdkBranch ==~ '[A-Za-z0-9][A-Za-z0-9._/-]*')) {
        error("Invalid Holoscan SDK branch: ${sdkBranch}")
    }

    def sdkRevision
    stage("Checkout Holoscan SDK ${sdkBranch}") {
        dir('holoscan-sdk') {
            deleteDir()
            checkout([
                $class: 'GitSCM',
                branches: [[name: "*/${sdkBranch}"]],
                extensions: [
                    [
                        $class: 'CloneOption',
                        depth: 0,
                        honorRefspec: false,
                        noTags: true,
                        shallow: false,
                        timeout: 30,
                    ],
                ],
                userRemoteConfigs: [[
                    credentialsId: Utils.get_sdk_credential(),
                    url: Utils.get_sdk_repository(),
                ]],
            ])
            withEnv(["SDK_CHECKOUT_DIR=${pwd()}"]) {
                sdkRevision = sh(
                    returnStdout: true,
                    script: '''
                        set -eu
                        git config --global --add safe.directory "$SDK_CHECKOUT_DIR"
                        git -c safe.directory="$SDK_CHECKOUT_DIR" rev-parse HEAD
                    ''',
                ).trim()
                if (!(sdkRevision ==~ /[0-9a-f]{40}/)) {
                    error("Could not resolve Holoscan SDK ${sdkBranch} revision")
                }
                echo("Holoscan SDK ${sdkBranch} revision: ${sdkRevision}")
            }
        }
    }
    return sdkRevision
}

def authenticate_sdk_build_cache() {
    def authenticated = false
    stage('Authenticate SDK build cache') {
        catchError(
            buildResult: 'SUCCESS',
            stageResult: 'UNSTABLE',
            catchInterruptions: false,
            message: 'SDK registry authentication failed; using a local container build.',
        ) {
            withEnv(["SDK_REGISTRY=${Utils.get_sdk_registry()}"]) {
                withCredentials([usernamePassword(
                    credentialsId: Utils.get_sdk_credential(),
                    usernameVariable: 'SDK_REGISTRY_USERNAME',
                    passwordVariable: 'SDK_REGISTRY_TOKEN',
                )]) {
                    def loginStatus = sh(
                        returnStatus: true,
                        script: '''
                            set +x
                            printf '%s' "$SDK_REGISTRY_TOKEN" |
                                docker login "$SDK_REGISTRY" \
                                    --username "$SDK_REGISTRY_USERNAME" \
                                    --password-stdin
                        ''',
                    )
                    if (loginStatus != 0) {
                        error("Docker login to ${Utils.get_sdk_registry()} failed")
                    }
                    authenticated = true
                }
            }
        }
        if (!authenticated) {
            echo('WARNING: Holoscan SDK registry cache unavailable; building the container locally.')
        }
    }
    return authenticated
}

def build_sdk(String sdkRevision, String sdkArchitecture) {
    def cacheImage = Utils.get_sdk_build_cache_image(sdkArchitecture, sdkRevision)
    def useRemoteCache = authenticate_sdk_build_cache()
    stage('Build Holoscan SDK') {
        dir('holoscan-sdk/public') {
            withEnv([
                "SDK_BUILD_CACHE_IMAGE=${cacheImage}",
                "USE_SDK_REMOTE_CACHE=${useRemoteCache}",
            ]) {
                sh '''
                set -eu

                if [ "$USE_SDK_REMOTE_CACHE" = true ]; then
                    echo "Attempting SDK build-container cache: $SDK_BUILD_CACHE_IMAGE"
                    if ! ./run build_image \
                        --cache-from "$SDK_BUILD_CACHE_IMAGE"; then
                        echo 'WARNING: Remote SDK cache build failed; retrying locally.' >&2
                        ./run build_image
                    fi
                else
                    ./run build_image
                fi

                ./run build \
                    --standalone \
                    --build-python false \
                    --build-benchmarks false \
                    --config "-DHOLOSCAN_BUILD_EXAMPLES:BOOL=OFF" \
                    --config "-DHOLOSCAN_BUILD_TESTS:BOOL=OFF"
            '''
            }
        }
    }
}

def resolve_sdk_install(
    String sdkRevision,
    String hostArchitecture,
    String sdkSource = 'pinned'
) {
    def sdkInstall = null
    stage('Resolve SDK installation') {
        def installName = sh(
            returnStdout: true,
            script: 'cd holoscan-sdk/public && ./run get_install_dir',
        ).trim()
        sdkInstall = "${pwd()}/holoscan-sdk/public/${installName}"
        if (!fileExists(
            "${sdkInstall}/lib/cmake/holoscan/holoscan-config.cmake"
        )) {
            error("Invalid SDK installation: ${sdkInstall}")
        }
        writeFile(
            file: 'ci-sdk-details.txt',
            text: (
                "revision=${sdkRevision}\n" +
                "source=${sdkSource}\n" +
                "architecture=${hostArchitecture}\n" +
                "install=${sdkInstall}\n"
            ),
        )
    }
    return sdkInstall
}

def cdash_build_name(Map flowSettings, String sourceBranch) {
    def architecture = flowSettings.cdash_arch?.trim()
    def sdkTrack = flowSettings.cdash_sdk?.trim()
    def branch = sourceBranch?.trim()
    if (!architecture || !sdkTrack || !branch) {
        throw new IllegalArgumentException(
            'CDash build names require cdash_arch, cdash_sdk, and source branch',
        )
    }
    return "holoscan-camera_${architecture}_sdk-${sdkTrack}_${branch}"
}

def build_module(Map flowSettings, boolean submitToCdash, boolean nightlyBuild) {
    stage('Build module') {
        def buildNameSuffix = env.gitlabSourceBranch?.trim() ?: 'main'
        def cdashBuildName = cdash_build_name(flowSettings, buildNameSuffix)
        withEnv([
            'CONTAINER_BUILD_LOG=ci-container-build.log',
            'CONTAINER_BUILD_EXIT_CODE_FILE=ci-container-build.exit-code',
            "HOLOSCAN_CAMERA_WRAPPER=${module_cli_wrapper()}",
            "CDASH_BUILD_NAME=${cdashBuildName}",
            "CDASH_MODEL=${nightlyBuild ? 'Nightly' : 'Experimental'}",
        ]) {
            try {
                sh '''#!/usr/bin/env bash
            set -uo pipefail
            "$HOLOSCAN_CAMERA_WRAPPER" build holoscan_camera_v4l2 \
                --local-sdk-root "$HOLOSCAN_SDK_INSTALL_DIR" \
                2>&1 | tee "$CONTAINER_BUILD_LOG"
            status=$?
            printf '%s\n' "$status" > "$CONTAINER_BUILD_EXIT_CODE_FILE"
            exit "$status"
        '''
            } catch (Throwable buildError) {
                if (submitToCdash) {
                    def exitCode = fileExists(env.CONTAINER_BUILD_EXIT_CODE_FILE)
                        ? readFile(env.CONTAINER_BUILD_EXIT_CODE_FILE).trim()
                        : '1'
                    if (!(exitCode ==~ /[0-9]+/)) {
                        exitCode = '1'
                    }
                    withEnv(["CONTAINER_BUILD_EXIT_CODE=${exitCode}"]) {
                        try {
                            sh '''
                                set -eu
                                python3 ci/report_cdash_test_failure.py \
                                    --cdash-url \
                                    'http://cdash.nvidia.com/submit.php?project=Holoscan-Modules' \
                                    --build-name "$CDASH_BUILD_NAME" \
                                    --site "Blossom-$(uname -m)" \
                                    --dashboard-model "$CDASH_MODEL" \
                                    --test-name 'holoscan_camera_v4l2.container_build' \
                                    --command "$HOLOSCAN_CAMERA_WRAPPER build holoscan_camera_v4l2" \
                                    --exit-code "$CONTAINER_BUILD_EXIT_CODE" \
                                    --log "$CONTAINER_BUILD_LOG" \
                                    --output build/cdash-fallback/Test.xml
                            '''
                        } catch (Throwable reportError) {
                            echo(
                                'WARNING: Failed to report the container-build ' +
                                "failure to CDash: ${reportError}"
                            )
                        }
                    }
                }
                throw buildError
            }
        }
    }
}

def test_module(Map flowSettings, boolean submitToCdash, boolean nightlyBuild) {
    stage('Test module') {
        def buildNameSuffix = env.gitlabSourceBranch?.trim() ?: 'main'
        def cdashBuildName = cdash_build_name(flowSettings, buildNameSuffix)
        withEnv([
            "HOLOSCAN_CAMERA_WRAPPER=${module_cli_wrapper()}",
            "CDASH_SUBMIT=${submitToCdash}",
            "CDASH_MODEL=${nightlyBuild ? 'Nightly' : 'Experimental'}",
            "CDASH_BUILD_NAME=${cdashBuildName}",
        ]) {
            sh '''
                set -eu
                if [ "$CDASH_SUBMIT" = true ]; then
                    set -- \
                        '--cdash-url=http://cdash.nvidia.com/submit.php?project=Holoscan-Modules' \
                        "--site-name=Blossom-$(uname -m)" \
                        "--ctest-options=-DDASHBOARD_MODEL=$CDASH_MODEL" \
                        "--ctest-options=-DCDASH_BUILD_NAME=$CDASH_BUILD_NAME"
                else
                    set --
                fi

                "$HOLOSCAN_CAMERA_WRAPPER" test holoscan_camera_v4l2 \
                    --language cpp \
                    --no-docker-build \
                    --no-xvfb \
                    "$@"
            '''
        }
    }
}

def package_module() {
    stage('Package module') {
        withEnv(["HOLOSCAN_CAMERA_WRAPPER=${module_cli_wrapper()}"]) {
            sh '''
            set -eu
            "$HOLOSCAN_CAMERA_WRAPPER" package holoscan-camera \
                --pkg-generator DEB \
                --local-sdk-root "$HOLOSCAN_SDK_INSTALL_DIR"
        '''
        }
    }
}

def collect_artifacts(String flowName) {
    sh(
        script: """
            set +e
            artifact_dir="ci-artifacts/${flowName}"
            mkdir -p "\$artifact_dir"
            find build -type f \\( -name LastTest.log -o -name junit.xml \\) \
                -exec cp {} "\$artifact_dir/" \\; 2>/dev/null
            find holoscan-sdk/public -type f -name sccache-stats.txt \
                -exec cp {} "\$artifact_dir/" \\; 2>/dev/null
            if [ -f ci-sdk-details.txt ]; then
                cp ci-sdk-details.txt "\$artifact_dir/"
            fi
            if [ -f ci-container-build.log ]; then
                cp ci-container-build.log "\$artifact_dir/"
            fi
            if [ -f build/cdash-fallback/Test.xml ]; then
                cp build/cdash-fallback/Test.xml \
                    "\$artifact_dir/cdash-container-build-Test.xml"
            fi
            find . -maxdepth 1 -type f -name '*.deb' \
                -exec cp {} "\$artifact_dir/" \\; 2>/dev/null
        """,
    )
    junit(
        allowEmptyResults: true,
        keepLongStdio: true,
        testResults: 'build/**/junit.xml',
    )
    archiveArtifacts(
        allowEmptyArchive: true,
        artifacts: "ci-artifacts/${flowName}/**",
        fingerprint: true,
    )
}

return this
