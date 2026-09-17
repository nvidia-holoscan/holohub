#!/usr/bin/env groovy
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

class FakeBuild {
    boolean nightly

    List getBuildCauses(String causeClass) {
        return nightly ? [causeClass] : []
    }
}

def load_utils(Map envValues = [:], boolean nightly = false) {
    def binding = new Binding([
        currentBuild: new FakeBuild(nightly: nightly),
        env: envValues,
    ])
    return new GroovyShell(this.class.classLoader, binding).evaluate(
        new File('ci/utils.groovy')
    )
}

def manualCases = [
    'rebuild',
    'Rebuild',
    'REBUILD',
    'rebuild [push]',
    'Rebuild [push]',
    'rebuild \\[push\\]',
    'rebuild because runner was unavailable',
    "rebuild\n[push]",
]
manualCases.each { comment ->
    def utils = load_utils([gitlabTriggerPhrase: comment])
    assert utils.is_manual_rebuild() : "expected manual rebuild for '${comment}'"
    assert utils.is_manual_rebuild_comment(comment) :
        "expected manual rebuild helper for '${comment}'"
}

def nonManualCases = [
    null,
    '',
    'rerun',
    'please rebuild',
]
nonManualCases.each { comment ->
    def utils = load_utils([gitlabTriggerPhrase: comment])
    assert !utils.is_manual_rebuild() : "expected non-manual rebuild for '${comment}'"
    assert !utils.is_manual_rebuild_comment(comment) :
        "expected non-manual rebuild helper for '${comment}'"
}

def pushCases = [
    'rebuild [push]',
    'rebuild \\[push\\]',
    'please [push] this',
]
pushCases.each { comment ->
    def utils = load_utils([gitlabTriggerPhrase: comment])
    assert utils.push_requested() : "expected push request for '${comment}'"
    assert utils.push_requested_comment(comment) :
        "expected push request helper for '${comment}'"
}

def nonPushCases = [
    null,
    '',
    'rebuild',
    'rebuild [Push]',
]
nonPushCases.each { comment ->
    def utils = load_utils([gitlabTriggerPhrase: comment])
    assert !utils.push_requested() : "expected no push request for '${comment}'"
    assert !utils.push_requested_comment(comment) :
        "expected no push request helper for '${comment}'"
}

def draftAuto = load_utils([
    gitlabMergeRequestIid: '3',
    gitlabMergeRequestTitle: 'Draft: CI: Report results',
    gitlabTriggerPhrase: null,
])
assert draftAuto.is_merge_request_build()
assert draftAuto.is_draft_merge_request()
assert !draftAuto.is_manual_rebuild()
assert draftAuto.is_merge_request_build() &&
    draftAuto.is_draft_merge_request() &&
    !draftAuto.is_manual_rebuild()
assert draftAuto.should_skip_draft_merge_request()

def draftPush = load_utils([
    gitlabMergeRequestIid: '3',
    gitlabMergeRequestTitle: 'Draft: CI: Report results',
    gitlabTriggerPhrase: 'rebuild [push]',
])
assert draftPush.is_merge_request_build()
assert draftPush.is_draft_merge_request()
assert draftPush.is_manual_rebuild()
assert !(draftPush.is_merge_request_build() &&
    draftPush.is_draft_merge_request() &&
    !draftPush.is_manual_rebuild())
assert !draftPush.should_skip_draft_merge_request()

assert !load_utils([:]).is_nightly_build()
assert load_utils([:], true).is_nightly_build()
def flowExcludedNode = '2u1g-b650-1788.ipp3a2.colossus'
def schedulingUtils = load_utils([BLACKLISTED_NODES: 'bad-a, bad-b'])
assert schedulingUtils.get_excluded_nodes([:]) == ['bad-a', 'bad-b', flowExcludedNode]
assert schedulingUtils.get_excluded_nodes([exclude_nodes: 'bad-b, bad-c']) ==
    ['bad-a', 'bad-b', flowExcludedNode, 'bad-c']
assert schedulingUtils.get_excluded_nodes([exclude_nodes: ['bad-c', 'bad-d']]) ==
    ['bad-a', 'bad-b', flowExcludedNode, 'bad-c', 'bad-d']
assert load_utils([:]).get_excluded_nodes([:]) == [flowExcludedNode]
def sdkUtils = load_utils([:])
def sdkRevision = '0123456789abcdef0123456789abcdef01234567'
assert sdkUtils.get_sdk_registry() == 'gitlab-master.nvidia.com:5005'
assert sdkUtils.get_sdk_build_cache_image('x86_64', sdkRevision) ==
    'gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/build-x86_64:012345678'
assert sdkUtils.get_sdk_build_cache_image('aarch64', sdkRevision) ==
    'gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/build-aarch64:012345678'
def schedulingYaml = schedulingUtils.get_pod_yaml([
    kubernetes_arch: 'amd64',
    container_name: 'tester',
    cpus: 13,
    memory: '60Gi',
    ephemeral_storage: '200Gi',
    gpus: 1,
    exclude_nodes: ['bad-c'],
])
assert schedulingYaml.contains('key: kubernetes.io/hostname')
assert schedulingYaml.contains('key: kubernetes.io/hostname\n                operator: NotIn')
assert schedulingYaml.contains('- "bad-a"')
assert schedulingYaml.contains('- "2u1g-b650-1788.ipp3a2.colossus"')
assert schedulingYaml.contains('key: nvidia.com/driver_version')
load_utils([:]).validate_comment_trigger_helpers()

println 'ci/utils.groovy tests passed'
