/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Credit for this code: https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/llamaspeak/static/websocket.js
 *
 * handles all the websocket streaming of text and audio to/from the server
 */

var websocket;
var msg_count_rx;
var msg_count_tx = 0;

function initAudioContext() {
    if (window.audioContext) {
        return; // AudioContext already initialized
    }
    try {
        window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('AudioContext initialized successfully in websocket.js');
        console.log('Audio State:', window.audioContext.state);
        console.log('Sample Rate:', window.audioContext.sampleRate);

        window.audioContext.onstatechange = function() {
            console.log('AudioContext state changed to', window.audioContext.state);
        };

        if (navigator.mediaDevices) {
            navigator.mediaDevices.addEventListener('devicechange', updateAudioDevices);
        }
    } catch (e) {
        console.error('Failed to create AudioContext:', e);
    }
}

async function updateAudioDevices() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioDevices = devices.filter(device => device.kind === 'audiooutput');
        console.log('Available audio output devices:', audioDevices);
        
        if (window.audioContext && window.audioContext.state === 'suspended') {
            await window.audioContext.resume();
        }
    } catch (error) {
        console.error('Error updating audio devices:', error);
    }
}

function reportError(msg) {
    console.error(msg);
}

function getWebsocketProtocol() {
    return window.location.protocol === 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(port = 49000) {
    return `${getWebsocketProtocol()}${window.location.hostname}:${port}/`;
}

function checkAudioSupport() {
    if (!window.AudioContext && !window.webkitAudioContext) {
        console.error('Web Audio API is not supported in this browser');
        return false;
    }
    return true;
}

function sendWebsocket(payload, type = 0) {

    if (websocket.readyState === WebSocket.OPEN) {
    const timestamp = Date.now();	
    console.log('Sending websocket message', type, payload);
    let header = new DataView(new ArrayBuffer(32));
    
    header.setBigUint64(0, BigInt(msg_count_tx));
    header.setBigUint64(8, BigInt(timestamp));
    header.setUint16(16, 42);
    header.setUint16(18, type);
    
    msg_count_tx++;
    
    let payloadSize;
    
    if (payload instanceof ArrayBuffer || ArrayBuffer.isView(payload)) {
        payloadSize = payload.byteLength;
    } else if (payload instanceof Blob) {
        payloadSize = payload.size;
    } else {
        payload = new TextEncoder().encode(JSON.stringify(payload));
        payloadSize = payload.byteLength;
    }
    
    header.setUint32(20, payloadSize);

    websocket.send(new Blob([header, payload]));}
    else {
        console.error('WebSocket is not open. ReadyState:', websocket.readyState);
    }

}

function onWebsocket(event) {
    const msg = event.data;
    
    if (msg.size <= 32) {
        console.log(`Received invalid websocket msg (size=${msg.size})`);
        return;
    }

    const header = msg.slice(0, 32);
    const payload = msg.slice(32);

    header.arrayBuffer().then((headerBuffer) => {
        const view = new DataView(headerBuffer);

        const msg_id = Number(view.getBigUint64(0));
        const timestamp = view.getBigUint64(8);
        const magic_number = view.getUint16(16);
        const msg_type = view.getUint16(18);
        const payload_size = view.getUint32(20);

        if (magic_number !== 42) {
            console.log(`Received invalid websocket msg (magic_number=${magic_number}, size=${msg.size})`);
        }

        if (payload_size !== payload.size) {
            console.log(`Received invalid websocket msg (payload_size=${payload_size}, actual=${payload.size})`);
        }

        if (msg_count_rx !== undefined && msg_id !== (msg_count_rx + 1)) {
            console.log(`Warning: Out-of-order message ID ${msg_id} (last=${msg_count_rx})`);
        }

        msg_count_rx = msg_id;

        if (msg_type === 0) { // JSON message
            payload.text().then((text) => {
                let json = JSON.parse(text);
                if ('chat_history' in json) {
                    updateChatHistory(json.chat_history);
                }
                if ('tegrastats' in json) {
                    console.log(json.tegrastats);
                }
            });
        } else if (msg_type === 1) { // TEXT message
            payload.text().then((text) => {
                console.log(`Text message: ${text}`);
            });
        } else if (msg_type === 2) { // AUDIO message
            payload.arrayBuffer().then((payloadBuffer) => {
                playAudio(payloadBuffer);
            });
        }
    });
}

function playAudio(audioData) {
    if (!window.audioContext) {
        console.warn('AudioContext not initialized. Initializing now.');
        initAudioContext();
    }
    
    if (!window.audioContext) {
        console.error('AudioContext not available');
        return;
    }
    
    window.audioContext.decodeAudioData(audioData, (buffer) => {
        const source = window.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(window.audioContext.destination);
        source.start(0);
    }, (err) => console.error('Error decoding audio data', err));
}

function updateChatHistory(chat_history) {
    var chc = document.getElementById('chat-history-container');
    var isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;

    $('#chat-history-container').empty();

    for (let n = 0; n < chat_history.length; n++) {
        for (let m = 0; m < chat_history[n].length; m++) {
            if (chat_history[n][m].length > 0) {
                $('#chat-history-container').append(
                    `<div id="msg_${n}_${m}" class="chat-message-user-${m} mb-3">${chat_history[n][m]}</div><br/>`
                );
            }
            let lastAddedMessage = $(`#msg_${n}_${m}`);
            if (lastAddedMessage.length > 0) {
                let images = lastAddedMessage.find('img');
                if (images.length > 0) {
                    images.on('load', function() {
                        if (isScrolledToBottom) 
                            chc.scrollTop = chc.scrollHeight - chc.clientHeight;
                    });
                }
            }
        }
    }

    if (isScrolledToBottom) {
        chc.scrollTop = chc.scrollHeight - chc.clientHeight;
    }
}

function connectWebsocket() {
    var websocketPortElement = document.getElementById('websocket-port');
    var websocketPort = websocketPortElement ? parseInt(websocketPortElement.getAttribute('data-port'), 10) : 49000;
    console.log('Connecting to WebSocket on port:', websocketPort);
    websocket = new WebSocket(getWebsocketURL(websocketPort));
    websocket.addEventListener('message', onWebsocket);
    websocket.addEventListener('open', function(event) {
        console.log('WebSocket connection established');
    });
    websocket.addEventListener('error', function(event) {
        console.error('WebSocket error:', event);
    });
}

function handleUserInteraction() {
    if (!window.audioContext) {
        initAudioContext();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    connectWebsocket();
    document.body.addEventListener('click', handleUserInteraction, { once: true });
});

console.log('websocket.js loaded');