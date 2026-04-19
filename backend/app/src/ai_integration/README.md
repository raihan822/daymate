# First setup in the llm_config with your desired models then go to your python script and use like below:

You are gonna use two methods, 
1. msg=prompt_for_model(prompt, system_instruction) then,
2. llm=load_llm_objects['model_name']
    - llm.invoke(msg).content


## Code Example use:
```python
    # 1. AI Logic
    # First set the llm_config then come back here.
    # RAG system (Ai-model) -> load_model[NAME].invoke(Message with Prompt).content:
    prompt = (
        f"Build your prompts.. {data}. "
        "prompts continue........................... etc!"
    )
    message = prompt_for_model(
        prompt=prompt, 
        system_instruction="You are a Doctor, give helpful medical advice."
        )
    
    # 3. Calling AI Model:--
    llm = load_llm_objects['model_key_in_llm_config']   #load_llm_objects itself is a dictionary but dynamic pydantic dictionary. just give the key wich model to prepare and it will do so.
    if llm:
        print("\nLLM Api key is Found\n")
        response_obj = llm.invoke(message)

        #response_text = response_obj.content[0].get('text', 'No text returned') #if Google's gemini model
        if isinstance(response_obj.content, list):  #gemini model onek shomoy list of dict['text'] e response dae
                    response_text = response_obj.content[0].get('text', 'No text returned')
                else:
                    response_text = response_obj.content
        return {"planning": response_text, "prompt": prompt}
    else:
        #fallback logic
        print("llm model not available")
```