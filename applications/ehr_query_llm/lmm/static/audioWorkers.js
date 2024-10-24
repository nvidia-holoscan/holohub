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
 * https://web.dev/patterns/media/microphone-process/
 * https://gist.github.com/flpvsk/047140b31c968001dc563998f7440cc1
 * https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor
 * https://stackoverflow.com/questions/57583266/audioworklet-set-output-to-float32array-to-stream-live-audio
 * https://github.com/GoogleChromeLabs/web-audio-samples/tree/main/src/audio-worklet/design-pattern/wasm-ring-buffer
 */


class AudioCaptureProcessor extends AudioWorkletProcessor {
    process([inputs], [outputs], parameters) {
          //console.log(`AudioCaptureProcessor::process(${inputs.length}, ${outputs.length})`);
          // convert float->int16 samples
          var input = inputs[0]; 
          var samples = new Int16Array(input.length);
  
          for( let i=0; i < input.length; i++ )
              samples[i] = 32767 * Math.min(Math.max(input[i], -1), 1);
          
          this.port.postMessage(samples, [samples.buffer]);
          
          // relay outputs
          //for( let i=0; i < inputs.length && i < outputs.length; i++ )
          //	outputs[i].set(inputs[i]);
          
      return true;
    }
  }
  
  class AudioOutputProcessor extends AudioWorkletProcessor {
      constructor(options) {
          super();
  
          this.queue = [];
          this.playhead = 0;
  
          this.port.onmessage = this.onmessage.bind(this);
      }
      
      onmessage(event) {
          const { data } = event;
          this.queue.push(data);
      }
      
      process([inputs], [outputs], parameters) {
          const output = outputs[0];
          var samplesWritten = 0;
          
          /*for(let i = 0; i < output.length; i++) {
                  output[i] = Math.sin(this.sampleCount * (Math.sin(this.sampleCount/24000.0) + 440.0) * Math.PI * 2.0 / 48000.0); //* 32767;
                  this.sampleCount++;
          }*/
          
          //console.log(`audio queue length ${this.queue.length} ${output.length} ${this.playhead}`);
          
          while( this.queue.length > 0 && samplesWritten < output.length ) {
              for( let i=samplesWritten; i < output.length && this.playhead < this.queue[0].length; i++ ) {
                  output[i] = this.queue[0][this.playhead] / 32767.0;
                  this.playhead++;
                  samplesWritten++;
              }
              
              if( this.playhead >= this.queue[0].length ) {
                  this.queue.shift();
                  this.playhead = 0;
              }
          }
  
          /*if( samplesWritten < output.length ) {
              console.warn(`gap in output audio  (${samplesWritten} of ${output.length} samples written)`);
          }*/
          
          for( let i=samplesWritten; i < output.length; i++ ) {
              output[i] = 0;
          }
          
          for( let i=1; i < outputs.length; i++ )
              outputs[i].set(outputs[0]);
          
      return true;
    }
  }
  
  registerProcessor("AudioCaptureProcessor", AudioCaptureProcessor);
  registerProcessor("AudioOutputProcessor", AudioOutputProcessor);