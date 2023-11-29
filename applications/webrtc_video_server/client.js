/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// peer connection
var pc = null;


async function getIceServers() {
    return fetch('/iceServers',
    ).then(function (response) {
        return response.json();
    }).catch(function (e) {
        alert(e);
    });
}

function createPeerConnection(iceServers) {
    var config = {
        sdpSemantics: 'unified-plan',
        iceServers: iceServers,
    };


    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', function (evt) {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function (offer) {
        return pc.setLocalDescription(offer);
    }).then(function () {
        var offer = pc.localDescription;

        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function (response) {
        return response.json();
    }).then(function (answer) {
        return pc.setRemoteDescription(answer);
    }).then(function () {
        // wait for ICE gathering to complete
        return new Promise(function (resolve) {
            if (pc.iceGatheringState === 'complete') {
                console.log("Ice gathering complete")
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        console.log("Ice gathering complete")
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).catch(function (e) {
        alert(e);
    });
}

async function start() {
    document.getElementById('start').style.display = 'none';

    var iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }]
    var newIceServers = await getIceServers();
    iceServers = iceServers.concat(newIceServers)
    console.log("Using the following ice servers: " + JSON.stringify(iceServers));

    pc = createPeerConnection(iceServers);

    pc.addTransceiver('video', { direction: 'recvonly' });

    negotiate();

    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function (transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close peer connection
    setTimeout(function () {
        pc.close();
    }, 500);

    document.getElementById('start').style.display = 'inline-block';
}
