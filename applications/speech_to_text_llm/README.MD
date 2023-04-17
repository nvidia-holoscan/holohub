## Speech-to-text + Large Language Model

This application transcribes an audio file using a speech-to-text model (STT), then uses a large language model (LLM) to summarize and generate new relevant information. 

While this workflow in principle could be used for a number of domains, here we provide a healthcare specific example. A `sample.wav` file is provided which is an example of a radiology interpretation. An OpenAI whisper model is used to transform the audio into text, then an API call is made to either the GPT3.5-turbo or GPT4 LLM. 

### YAML Configuration

The input (either audio or video file), specific Whisper model (tiny, small, medium, or large), LLM model, and directions for the LLM are all determined by the `stt_to_nlp.yaml` file. As you see from our example, the directions for the LLM are made via natural language, and can result in very different applications. 

For our purposes we specify the directions as:

```
  context: 'Make summary of the transcript (and correct any transcription errors in CAPS).\n Create a Patient Summary with no medical jargon. \n 
  Create a full radiological report write-up. \n Give likely ICD-10 Codes \n Suggested follow-up steps.'
```

This results in the following output from the LLM:

```
LLM Response: 
 Summary of Transcript:
The patient has full thickness wear on the dorsal half of the second metatarsal head with reactive bone marrow edema and capsulitis. There is also second web space bursitis and a third web space neuroma. The 51-year-old male has multiple gallbladder polyps, with the largest measuring 1.9 x 2 cm, 1.7 x 1.7 cm in the mid portion, and 1.6 x 1.6 cm distally. Other smaller polyps are also present.

Patient Summary (No Medical Jargon):
The patient has damage and inflammation in the foot, specifically in the second toe joint and surrounding areas. They also have multiple growths in their gallbladder, with the largest being about the size of a grape. The patient is a 51-year-old male with a family history of abdominal aortic aneurysm.

Full Radiological Report Write-up:
Patient: 51-year-old male
Family History: Abdominal aortic aneurysm

Findings:
1. Foot: Full thickness wear over the dorsal half of the second metatarsal head with reactive subchondral bone marrow edema and capsulitis. Second web space intermetatarsal bursitis and a third web space neuroma are also present.
2. Gallbladder: Multiple gallbladder polyps are observed. The largest polyp measures 1.9 x 2 cm, with additional polyps measuring 1.7 x 1.7 cm in the mid portion and 1.6 x 1.6 cm distally. Two smaller polyps measure 0.5 x 0.4 x 0.4 cm and 0.5 x 0.3 x 0.5 cm.

Likely ICD-10 Codes:
1. M25.572 - Capsulitis, left ankle and foot
2. M79.671 - Bursitis, right ankle and foot
3. G57.60 - Lesion of plantar nerve, unspecified lower limb
4. K82.8 - Other specified diseases of the gallbladder (gallbladder polyps)

Suggested Follow-up Steps:
1. For the foot issues, the patient may benefit from a consultation with a podiatrist or orthopedic specialist to discuss treatment options, which may include physical therapy, orthotics, or surgery.
2. For the gallbladder polyps, the patient should consult with a gastroenterologist to determine the need for further evaluation, monitoring, or possible surgical intervention. Regular ultrasound examinations may be recommended to monitor the growth of the polyps.
```

### Run Instructions

Note: To run this application you will need to [create an OpenAI account](https://platform.openai.com/signup) and obtain your own [API key](https://platform.openai.com/account/api-keys) with active credits.

* (Optional) Create and use a virtual environment:

  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* Install the python packages

  ```
  pip install -r applications/speech_to_text_llm/requirements.txt
  ```

* Run the application

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
  cd applications/speech_to_text_llm 
  python3 stt_to_nlp.py
  ```
### Sample Audio File

Please note the sample audio file included is licensed as CC-BY-4.0 International, copyright NVIDIA 2023. 
