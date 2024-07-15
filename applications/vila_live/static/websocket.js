/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
var msg_count_tx=0;

function reportError(msg) {
  console.log(msg);
}
 
function getWebsocketProtocol() {
  return window.location.protocol == 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(port=49000) {  // wss://192.168.1.2:49000
  return `${getWebsocketProtocol()}${window.location.hostname}:${port}/${name}`;
}

function sendWebsocket(payload, type=0) {
  const timestamp = Date.now();	
	let header = new DataView(new ArrayBuffer(32));
		
	header.setBigUint64(0, BigInt(msg_count_tx));
	header.setBigUint64(8, BigInt(timestamp));
	header.setUint16(16, 42);
	header.setUint16(18, type);
	
	msg_count_tx++;
	
	let payloadSize;
	
	if( payload instanceof ArrayBuffer || ArrayBuffer.isView(payload) ) { // binary
		payloadSize = payload.byteLength;
	}
	else if( payload instanceof Blob) {
		payloadSize = payload.size;
	}
	else { // serialize to JSON
		payload = new TextEncoder().encode(JSON.stringify(payload)); // Uint8Array
		payloadSize = payload.byteLength;
	}
	
	header.setUint32(20, payloadSize);
	
	websocket.send(new Blob([header, payload]));
}

function onWebsocket(event) {
	const msg = event.data;
	
	if( msg.size <= 32 ) {
		console.log(`received invalid websocket msg (size=${msg.size})`);
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
		
		if( magic_number != 42 ) {
			console.log(`received invalid websocket msg (magic_number=${magic_number}  size=${msg.size}`);
		}
		
		if( payload_size != payload.size ) {
			console.log(`received invalid websocket msg (payload_size=${payload_size} actual=${payload.size}`);
		}
		
		if( msg_count_rx != undefined && msg_id != (msg_count_rx + 1) )
			console.log(`warning:  out-of-order message ID ${msg_id}  (last=${msg_count_rx})`);
			
		msg_count_rx = msg_id;
		
		if( msg_type == 0 ) { // JSON message
			payload.text().then((text) => {
				json = JSON.parse(text);
				
				if( 'chat_history' in json ) {
					const chat_history = json['chat_history'];
					
					var chc = document.getElementById('chat-history-container');
					var isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;
					
					$('#chat-history-container').empty(); // started clearing because server may remove partial/rejected ASR prompts
					
					for( let n=0; n < chat_history.length; n++ ) {
						for( let m=0; m < chat_history[n].length; m++ ) {
							prev_msg = $(`#chat-history-container #msg_${n}_${m}`);
							if( prev_msg.length > 0 ) {
								prev_msg.html(chat_history[n][m]);
							}
							else if(chat_history[n][m].length > 0 ) {
								$('#chat-history-container').append(
									`<div id="msg_${n}_${m}" class="chat-message-user-${m} mb-3">${chat_history[n][m]}</div><br/>`
								);
							}
						}
					}
					
					if( isScrolledToBottom ) // autoscroll unless the user has scrolled up
						chc.scrollTop = chc.scrollHeight - chc.clientHeight;
				} else if( 'image_b64' in json ) {
					const image_b64 = json['image_b64'];
					const img = document.getElementById('image');
					const b64_src = 'data:image/jpeg;base64,' + image_b64;
					img.src = b64_src;
				}
				
				if( 'tegrastats' in json ) {
					console.log(json['tegrastats']);
				}
			});
		}
		if( msg_type == 1 ) { // TEXT message
			payload.text().then((text) => {
				console.log(`text message: ${text}`);
			});
		}
		else if( msg_type == 2 ) { // AUDIO message
			payload.arrayBuffer().then((payloadBuffer) => {
				onAudioOutput(payloadBuffer);
			});
		}
	});
}

function connectWebsocket() {
	websocket = new WebSocket(getWebsocketURL());
	websocket.addEventListener('message', onWebsocket);
}
