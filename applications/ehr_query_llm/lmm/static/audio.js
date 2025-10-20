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
 * Credit for this code: https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/llamaspeak/static/audio.js
 * 
 * flask webserver
 * handles audio input/output device streaming
 * websocket.js should be included also
 */

var audioInputTrack;    // MediaStreamTrack
var audioInputDevice;   // MediaStream
var audioInputStream;   // MediaStreamAudioSourceNode
var audioInputCapture;  // AudioWorkletNode
var audioOutputWorker;  // AudioWorkletNode
var audioOutputMuted = false;

function checkMediaDevices() {
  return (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !navigator.mediaDevices.enumerateDevices) ? false : true;
}

function enumerateAudioDevices() {
    var selectInput = document.getElementById('audio-input-select');
    var selectOutput = document.getElementById('audio-output-select');
    
    if (!checkMediaDevices()) {
        selectInput.add(new Option('use HTTPS to enable browser audio'));
        selectOutput.add(new Option('use HTTPS to enable browser audio'));
        return;
    }
    
    navigator.mediaDevices.getUserMedia({audio: true, video: false}).then((stream) => { // get permission from user
        navigator.mediaDevices.enumerateDevices().then((devices) => {
            stream.getTracks().forEach(track => track.stop()); // close the device opened to get permissions
            devices.forEach((device) => {
                console.log(`Browser media device:  ${device.kind}  label=${device.label}  id=${device.deviceId}`);
                
                if (device.kind == 'audioinput')
                    selectInput.add(new Option(device.label, device.deviceId));
                else if (device.kind == 'audiooutput')
                    selectOutput.add(new Option(device.label, device.deviceId));
            });
            
            if (selectInput.options.length == 0)
                selectInput.add(new Option('browser has no audio inputs available'));

            if (selectOutput.options.length == 0)
                selectOutput.add(new Option('browser has no audio outputs available'));
        });
    }).catch(reportError);
}

function visualizeAudio(audioContext, source) {
    const analyser = audioContext.createAnalyser();
    source.connect(analyser);
    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        const level = Math.max(...dataArray) - 128;
        console.log('Audio level:', level);
    }

    draw();
}
document.getElementById('initAudioButton').addEventListener('click', function() {
    console.log('Initializing audio devices');
    openAudioDevices();
});

function openAudioDevices(inputDeviceId, outputDeviceId) {
    console.log('Starting openAudioDevices');
    if (inputDeviceId == undefined || inputDeviceId == "Default")
        inputDeviceId = null;
    
    if (outputDeviceId == undefined || outputDeviceId == "Default")
        outputDeviceId = null;
    
    console.log(`Using input device: ${inputDeviceId}, output device: ${outputDeviceId}`);

    const constraints = {
        audio: inputDeviceId ? {deviceId: {exact: inputDeviceId}} : true,
        video: false
    };
    
    navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
        console.log('Opened audio input device', inputDeviceId);

        audioInputDevice = stream;
        audioInputTrack = stream.getAudioTracks()[0];
        audioSettings = audioInputTrack.getSettings();
        
        audioInputTrack.enabled = true;  // mic on by default
        
        console.log('Audio input track:', audioInputTrack);
        console.log('Audio settings:', audioSettings);
        
        // Ensure AudioContext is created here
        if (!window.audioContext) {
            try {
                window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log('AudioContext created:', window.audioContext);
            } catch (e) {
                console.error('Failed to create AudioContext:', e);
                return;
            }
        }
        
        audioInputStream = window.audioContext.createMediaStreamSource(audioInputDevice);
        console.log('Created media stream source');

        // Check if audioWorklet is supported
        if (window.audioContext.audioWorklet) {
            window.audioContext.audioWorklet.addModule("/static/audioWorkers.js").then(() => {
                console.log('AudioWorklet module loaded');
                audioInputCapture = new AudioWorkletNode(window.audioContext, "AudioCaptureProcessor");
                audioOutputWorker = new AudioWorkletNode(window.audioContext, "AudioOutputProcessor");
                audioInputStream.connect(audioInputCapture).connect(audioOutputWorker).connect(window.audioContext.destination);
                console.log('Audio processing chain set up');

                audioInputCapture.port.onmessage = onAudioInputCapture;
                
                // Add visualization
                visualizeAudio(window.audioContext, audioInputStream);
            }).catch(error => {
                console.error('Error loading AudioWorklet:', error);
            });
        } else {
            console.error('AudioWorklet is not supported in this browser');
            // Here you might want to implement a fallback method
        }
    })
    .catch(error => {
        console.error('Error accessing audio device:', error);
    });
    }



function onAudioInputCapture(event) {
        console.log('Audio input captured', event.data);
        if (audioInputTrack && audioInputTrack.enabled) {  // unmuted
            console.log('Sending audio data to websocket');
            sendWebsocket(event.data, type=2);  // event.data is a Uint16Array
        } else {
            console.log('Audio input is muted or track is not available');
        }
}

function onAudioOutput(samples) {
    if (audioOutputWorker != undefined && !audioOutputMuted) {
        int16Array = new Int16Array(samples);
        audioOutputWorker.port.postMessage(int16Array, [int16Array.buffer]);
    }
}

function muteAudioInput() {  
    var button = document.getElementById('audio-input-mute');
    const muted = button.classList.contains('bi-mic-fill');
    console.log(`muteAudioInput(${muted})`);
    if (muted)
        button.classList.replace('bi-mic-fill', 'bi-mic-mute-fill');
    else
        button.classList.replace('bi-mic-mute-fill', 'bi-mic-fill');
    if (audioInputTrack != undefined)
        audioInputTrack.enabled = !muted;
}

function muteAudioOutput() {  
    var button = document.getElementById('audio-output-mute');
    const muted = button.classList.contains('bi-volume-up-fill');
    console.log(`muteAudioOutput(${muted})`);
    if (muted)
        button.classList.replace('bi-volume-up-fill', 'bi-volume-mute-fill');
    else
        button.classList.replace('bi-volume-mute-fill', 'bi-volume-up-fill');
    audioOutputMuted = muted;
}

function handleUserInteraction() {
    console.log('User interaction detected');
    if (!window.audioContext) {
        console.log('Initializing audio devices');
        openAudioDevices();
    } else {
        console.log('AudioContext already initialized');
    }
}

document.addEventListener('DOMContentLoaded', function() {
    enumerateAudioDevices();
    document.body.addEventListener('click', handleUserInteraction, { once: true });
});

console.log('audio.js loaded');