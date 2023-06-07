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
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SIGNAL_PACKET_SIZE 8192
#define HEADER_SIZE 1
#define COMPLEX_SIZE 2
#define BUFFER_SIZE (HEADER_SIZE + (SIGNAL_PACKET_SIZE * COMPLEX_SIZE))
#define NUMBER_OF_BUFFERS 2

#define NUMBER_OF_ITERATIONS 15

int16_t main_buffer[BUFFER_SIZE*NUMBER_OF_BUFFERS];
bool valid_buffer[NUMBER_OF_BUFFERS];
pthread_mutex_t mutexen[NUMBER_OF_BUFFERS];
pthread_cond_t signal[NUMBER_OF_BUFFERS];
bool quit;

struct siggen_config{
  //Number of bits that are error
  unsigned int error_bits;
  // 0 to RAND_MAX
  unsigned int probability_of_transmission;
  unsigned int offset;
  unsigned int increment;
};

struct transmit_config{
  // UDP port
  unsigned short port;
  // IP address to transmit to
  char * address;
};

struct chirp_description{
  //0 to 2^15 - 1
  int16_t initial_delta;
  //0 to 2^31 - 1
  int32_t chirp_rate;
  // bits to attenuate signal by
  unsigned char attenuation;
  //Bin start
  unsigned long start;
  //Bin length
  unsigned long length;
};

struct live_signal{
  bool valid;
  struct chirp_description chirp;
};

/*
The fpsin and fpcos functions were from 
https://www.nullhardware.com/blog/fixed-point-sine-and-cosine-for-embedded-systems/
and have the following copywright:
==========LICENCE=============
Copyright 2018 Andrew Steadman

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://nam11.safelinks.protection.outlook.com/?url=http%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE-2.0&data=05%7C01%7Cjjomier%40nvidia.com%7C5992447d975048be2a7508db67525601%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638217375574205616%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=bHeZqmTlmFx5BCcketj0j%2FTod6eoeTOrkouAU%2FAojFY%3D&reserved=0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
==========END LICENCE==========

Implements the 5-order polynomial approximation to sin(x).
@param i   angle (with 2^15 units/circle)
@return    16 bit fixed point Sine value (4.12) (ie: +4096 = +1 & -4096 = -1)

The result is accurate to within +- 1 count. ie: +/-2.44e-4.
*/
int16_t fpsin(int16_t i)
{
    /* Convert (signed) input to a value between 0 and 8192. (8192 is pi/2, which is the region of the curve fit). */
    /* ------------------------------------------------------------------- */
    i = (int16_t)(i << 1);
    uint8_t c = i<0; //set carry for output pos/neg

    if(i == (i|0x4000)) // flip input value to corresponding value in range [0..8192)
        i = (int16_t)(((1<<15)) - i);
    i = (i & 0x7FFF) >> 1;
    /* ------------------------------------------------------------------- */

    /* The following section implements the formula:
     = y * 2^-n * ( A1 - 2^(q-p)* y * 2^-n * y * 2^-n * [B1 - 2^-r * y * 2^-n * C1 * y]) * 2^(a-q)
    Where the constants are defined as follows:
    */
    uint32_t A1=3370945099UL;
    uint32_t B1=2746362156UL;
    uint32_t C1=292421UL;
    unsigned char n=13;
    unsigned char p=32;
    unsigned char q=31;
    unsigned char r=3;
    unsigned char a=12;

    uint32_t y = ((uint32_t)C1*((uint32_t)i))>>n;
    y = B1 - (((uint32_t)i*y)>>r);
    y = (uint32_t)i * (y>>n);
    y = (uint32_t)i * (y>>n);
    y = A1 - (y>>(p-q));
    y = (uint32_t)i * (y>>n);
    y = (uint32_t)((y+(1UL<<(q-a-1)))>>(q-a)); // Rounding

    return (int16_t)(c ? -y : y);
}

//Cos(x) = sin(x + pi/2)
#define fpcos(i) fpsin((int16_t)(((uint16_t)(i)) + 8192U))


void linear_chirp_generator(int16_t * output, unsigned long length, const struct chirp_description * chirp)
{
  unsigned long end = (length < (chirp->start + chirp->length)) ? length : (chirp->start + chirp->length);
  int32_t delta = ((int32_t)chirp->initial_delta) << 16;
  uint16_t phase = 0;
  for(long unsigned int i=(chirp->start); i<end; i++){
    uint16_t real = (uint16_t)(fpcos((int16_t)phase) >> chirp->attenuation);
    uint16_t imag = (uint16_t)(fpsin((int16_t)phase) >> chirp->attenuation);
    output[i*2] = (int16_t)htons((uint16_t)(real + ntohs((uint16_t)output[i*2]))); //Real
    output[i*2 + 1] = (int16_t)htons((uint16_t)(imag + ntohs((uint16_t)output[i*2 + 1]))); //Real
    phase = (uint16_t)(phase + (delta >> 16));
    delta += chirp->chirp_rate;
  }
}

void fill_with_noise(int16_t * start, unsigned long length, unsigned int error_bits)
{
  for(unsigned long i=0;
      i<length;
      i++){
    int16_t rand_val = (int16_t)(rand() & 0xFFFF);
    uint16_t noise_val = (uint16_t)(rand_val % (1 << (error_bits)));
    start[i] = (int16_t)htons(noise_val);
  }
}

void * siggen_thread(void * ptr)
{
  struct siggen_config * config = (struct siggen_config*) ptr;
  uint16_t id = 0;

  fill_with_noise(main_buffer, BUFFER_SIZE*NUMBER_OF_BUFFERS, config->error_bits);

  struct live_signal signals[NUMBER_OF_BUFFERS];
  for(long i=0; i<NUMBER_OF_BUFFERS; i++){
    ((uint16_t *)main_buffer)[i*BUFFER_SIZE] = id++;
    signals[i].valid = false;
  }

  unsigned long count = 0;
  unsigned int index = config->offset;
  unsigned int inc = config->increment;
  
  struct live_signal next_signal;
  while(count < NUMBER_OF_ITERATIONS){
    index = count % NUMBER_OF_BUFFERS;
    count += inc;

    if(config->probability_of_transmission > (unsigned int)rand()){
      next_signal.valid = true;
      next_signal.chirp.start = 0;
      next_signal.chirp.length = SIGNAL_PACKET_SIZE;
      next_signal.chirp.initial_delta = -0x7000;
      next_signal.chirp.chirp_rate = -0x00004000;
      next_signal.chirp.attenuation = 1;
    } else {
      next_signal.valid = false;
    }

    pthread_mutex_lock(mutexen+index);
    while(valid_buffer[index]){
      if(quit){
        pthread_mutex_unlock(mutexen+index);
        return NULL;
      }
      pthread_cond_wait(signal+index, mutexen+index);
    }


    if(signals[index].valid){
      fill_with_noise(main_buffer+
            (BUFFER_SIZE*index)+
            HEADER_SIZE +
            (signals[index].chirp.start*COMPLEX_SIZE),
          signals[index].chirp.length*COMPLEX_SIZE,
          config->error_bits);
    }

    if(next_signal.valid){
      linear_chirp_generator(main_buffer + BUFFER_SIZE*index + HEADER_SIZE,
          BUFFER_SIZE,
          &next_signal.chirp);
    }

    valid_buffer[index] = true;
    pthread_cond_signal(signal+index);
    pthread_mutex_unlock(mutexen+index);
     
    signals[index] = next_signal;
  }
  quit = true;
  for(int i=0; i<NUMBER_OF_BUFFERS; i++){
    pthread_mutex_lock(mutexen+index);
    pthread_cond_signal(signal+index);
    pthread_mutex_unlock(mutexen+index);
  }
  return NULL;
}

void * transmit_thread(void * ptr)
{
  struct transmit_config * config = (struct transmit_config *)ptr;

  int sock = socket(
      AF_INET,
      SOCK_DGRAM,
      0);

  struct sockaddr_in dest_addr;
  dest_addr.sin_family = AF_INET;
  dest_addr.sin_port = htons(config->port);
  inet_pton(AF_INET, config->address, &dest_addr.sin_addr);

  ssize_t rc;

  unsigned long count = 0;
  unsigned int index = 0;
  while(1){
    index = count % NUMBER_OF_BUFFERS;
    count++;

    pthread_mutex_lock(mutexen + index);
    while(!(valid_buffer[index])){
      if(quit){
        pthread_mutex_unlock(mutexen+index);
        return NULL;
      }
      pthread_cond_wait(signal+index, mutexen+index);
    }

    rc = sendto(sock,
        (main_buffer + (BUFFER_SIZE*index)),
        BUFFER_SIZE * sizeof(main_buffer[0]),
        0,
        (struct sockaddr *)&dest_addr,
        sizeof(dest_addr));
    if(rc<0){
      perror("Failed to send");
      quit = true;
      for(unsigned int i=0; i<NUMBER_OF_BUFFERS; i++){
        if(i != index){
          pthread_mutex_lock(mutexen+index);
        }
        pthread_cond_signal(signal+index);
        pthread_mutex_unlock(mutexen+index);
      }
      return NULL;
    }

    valid_buffer[index] = false;
    pthread_cond_signal(signal+index);
    pthread_mutex_unlock(mutexen+index);
  }
}

int main(int argc, char *argv[])
{
  (void)argc;
  (void)argv;

  int rc;
  pthread_t siggen, transmit;

  struct siggen_config sg_config;
  struct transmit_config tx_config;

  sg_config.error_bits = 4;
  sg_config.probability_of_transmission = RAND_MAX >> 2;
  sg_config.increment = 1;
  sg_config.offset = 0;

  tx_config.port = 8999;
  tx_config.address = "127.0.0.1";
  srand(7);

  int opt;
  while((opt = getopt(argc, argv, "n:o:s:p:a:"))!= -1){
    switch(opt){
      case 'n':
        sg_config.increment = (unsigned int)atoi(optarg);
        break;
      case 'o':
        sg_config.offset = (unsigned int)atoi(optarg);
        break;
      case 's':
        srand((unsigned int)atoi(optarg));
        break;
      case 'p':
        tx_config.port = (short unsigned int)atoi(optarg);
        break;
      case 'a':
        tx_config.address = optarg;
        break;
      default:
        printf("Unrecognized command\n");
        return -1;
        break;
    }
  }

  quit = false;
  for(int i=0; i<NUMBER_OF_BUFFERS; i++){
    pthread_mutex_init(mutexen + i, NULL);
    pthread_cond_init(signal + i, NULL);
    valid_buffer[i]=false;
  }

  rc = pthread_create(&siggen, NULL, siggen_thread, &sg_config);
  if(rc){
    printf("Server thread creation failed\n");
    return rc;
  }
  rc = pthread_create(&transmit, NULL, transmit_thread, &tx_config);
  if(rc){
    printf("Client thread creation failed\n");
    return rc;
  }

  pthread_join(siggen, NULL);
  pthread_join(transmit, NULL);

  for(int i=0; i<NUMBER_OF_BUFFERS; i++){
    pthread_mutex_destroy(mutexen + i);
  }

  return 0;
}
