#!/usr/bin/env groovy
// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

class FakeBuild {
    boolean nightly
    boolean manual

    List getBuildCauses(String causeClass) {
        if (causeClass == 'hudson.triggers.TimerTrigger$TimerTriggerCause') {
            return nightly ? [causeClass] : []
        }
        if (causeClass == 'hudson.model.Cause$UserIdCause') {
            return manual ? [causeClass] : []
        }
        return []
    }
}

def load_utils(Map envValues = [:], boolean nightly = false, boolean manual = false) {
    def binding = new Binding([
        currentBuild: new FakeBuild(nightly: nightly, manual: manual),
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
assert !load_utils([:]).is_manual_build()
assert load_utils([:], false, true).is_manual_build()
load_utils([:]).validate_comment_trigger_helpers()

println 'ci/utils.groovy tests passed'
