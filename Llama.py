# !mkdir /tmp/submission

# %%time
import os,sys
os.system("pip install -U -t /tmp/submission/lib accelerate")
os.system("pip install -i https://pypi.org/simple/ -U -t /tmp/submission/lib bitsandbytes")
os.system("pip cache purge")
sys.path.insert(0, "/tmp/submission/lib")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "abacusai/Llama-3-Smaug-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quanty_type = "fp4", 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quanty = True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    torch_dtype = torch.float16,
    device_map = "auto",
    trust_remote_code = True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("/tmp/submission/weights")
tokenizer.save_pretrained("/tmp/submission/weights")

import pandas as pd
keywords = pd.read_csv("/kaggle/input/updated-kaggle-keywords/keywords_v2.csv")
keywords.head(1)

pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id

question = f"Is the secret word a country?"

keyword = "venezuela"
if keyword in keywords.keyword.values:
    row = keywords.loc[keywords.keyword==keyword].iloc[0]
    category = row.category #"landmark"
    continent = row.continent #"North America"
    negate = {
        "city":"Is is not a country. It is not a landmark.",
        "country":"Is is not a city. It is not a landmark.",
        "landmark":"Is is not a city. It is not a country.",
    }
    prompt = f"We are playing 20 questions. The keyword is {keyword}. It is a {category}. {negate[category]} This word has first letter {keyword[0]}. This {category} is located in {continent}. {question}"
else:
    prompt = f"We are playing 20 questions. The keyword is {keyword}. It is a thing. Is is not a city. It is not a country. It is not a landmark. This word has first letter {keyword[0]}. {question}"

messages = [
    {"role": "system", "content": "Answer yes or no to the following question and nothing else."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask = model_inputs.attention_mask,
    pad_token_id=pad_token_id,
    max_new_tokens=1
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
if not "yes" in response.lower(): response = "no"
else: response = "yes"

print(f"When keyword is '{keyword}'")
print(f"and questioner asks: '{question}'")
print(f"with prompt = {prompt}")
print(f"our model answers: '{response}'")

import gc, torch
del model, tokenizer, model_inputs, generated_ids, response
gc.collect()
torch.cuda.empty_cache()

'''
!cp /kaggle/input/updated-kaggle-keywords/keywords_v2.csv /kaggle/working
!cp /kaggle/input/updated-kaggle-keywords/keywords_v2.csv /tmp/submission
'''

# %%writefile /tmp/submission/main.py

################
# DETERMINE IF SUBMIT OR COMMIT
import os
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
VERBOSE = False
if not os.path.exists(KAGGLE_AGENT_PATH + "weights"):
    KAGGLE_AGENT_PATH = "/tmp/submission/"
    VERBOSE = True

#################
# LOAD MODEL INTO MEMORY
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, f"{KAGGLE_AGENT_PATH}lib")
model = AutoModelForCausalLM.from_pretrained(
    f"{KAGGLE_AGENT_PATH}weights/",
    torch_dtype = torch.float16,
    device_map = "auto",
    trust_remote_code = True,
)
tokenizer = AutoTokenizer.from_pretrained(f"{KAGGLE_AGENT_PATH}weights/")

##############
# BINARY SEARCH AS QUESTIONER
import pandas as pd, numpy as np
keywords = pd.read_csv(KAGGLE_AGENT_PATH + "keywords_v2.csv")
keywords['guess'] = 0

categories = ["city","country","landmark"]
#np.random.shuffle(categories)
category_yes = []
category_no = []
cat_guess = 0

continents = ["Europe","Asia","North America","Africa","South America","Australia"]
#np.random.shuffle(continents)
continent_yes = []
continent_no = []
con_guess = 0

first_letters = []
first_letter_yes = []
first_letter_no = []
let_guess = 0
extra_guess = ""

###############
# LLM MODEL AS ANSWERER
def get_yes_no(question,keyword):
    global keywords, VERBOSE
    
    if keyword in keywords.keyword.values:
        row = keywords.loc[keywords.keyword==keyword].iloc[0]
        category = row.category #"landmark"
        continent = row.continent #"North America"
        negate = {
            "city":"Is is not a country. It is not a landmark.",
            "country":"Is is not a city. It is not a landmark.",
            "landmark":"Is is not a city. It is not a country.",
        }
        prompt = f"We are playing 20 questions. The keyword is {keyword}. It is a {category}. {negate[category]} This word has first letter {keyword[0]}. This {category} is located in {continent}. {question}"
    else:
        prompt = f"We are playing 20 questions. The keyword is {keyword}. It is a thing. Is is not a city. It is not a country. It is not a landmark. This word has first letter {keyword[0]}. {question}"
    
    messages = [
        {"role": "system", "content": "Answer yes or no to the following question and nothing else."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask = model_inputs.attention_mask,
        pad_token_id = pad_token_id,
        max_new_tokens=1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if not "yes" in response.lower(): response = "no"
    else: response = "yes"
        
    if VERBOSE:
        print(f"### {prompt}")
        
    return response

############
# MAIN AGENT FUNCTION
def agent_fn(obs, cfg):
    global keywords, extra_guess, VERBOSE
    global categories, category_yes, category_no, cat_guess
    global continents, continent_yes, continent_no, con_guess
    global first_letters, first_letter_yes, first_letter_no, let_guess
        
    # GENERATE RESPONSE
    if obs.turnType == "ask":
        
        if (cat_guess<3)&(len(category_yes)==0):
            response = f"Is the keyword the name of a {categories[cat_guess]}?"
            cat_guess += 1
        elif (con_guess<6)&(len(continent_yes)==0):
            category = "place"
            if len( category_yes )==1: 
                category = category_yes[0]
            response = f"Is the {category} located in {continents[con_guess]}?"
            con_guess += 1
        else:
            IDX = keywords.category.isin( category_yes )
            IDX = IDX & (keywords.continent.isin( continent_yes ))
            first_letters = list(keywords.loc[IDX,"first_letter"].value_counts().index.values)
            if let_guess < len(first_letters):
                response = f"Does the keyword begin with the letter {first_letters[let_guess]}?"
            else:
                IDX = keywords.guess == 0
                if len(category_yes)>0: IDX = IDX & (keywords.category.isin(category_yes))
                if len(category_no)>0: IDX = IDX & (~keywords.category.isin(category_no))
                if len(continent_yes)>0: IDX = IDX & (keywords.continent.isin(continent_yes))
                if len(continent_no)>0: IDX = IDX & (~keywords.continent.isin(continent_no))
                if len(first_letter_yes)>0: IDX = IDX & (keywords.first_letter.isin(first_letter_yes))
                if len(first_letter_no)>0: IDX = IDX & (~keywords.first_letter.isin(first_letter_no))
                try:
                    guess = keywords.loc[IDX].sample(1).index.values[0]
                    keywords.loc[guess,'guess'] = 1
                    response = keywords.loc[guess,"keyword"]
                except:
                    response = np.random.choice( keywords.keyword.values )
                extra_guess = response
                response = f"Is it {response}?"
            let_guess += 1
            
    elif obs.turnType == "guess":
        
        category_yes = []
        category_no = []
        for k in range(cat_guess):
            if obs.answers[k]=="yes":
                category_yes.append( categories[k] )
            else:
                category_no.append( categories[k] )
        if (cat_guess==3)&(len(category_yes)==0):
            category_yes = ["city","country","landmark"]
            category_no = []
            
        continent_yes = []
        continent_no = []
        for k in range(con_guess):
            if obs.answers[k+cat_guess]=="yes":
                continent_yes.append( continents[k] )
            else:
                continent_no.append( continents[k] )
        if (con_guess==6)&(len(continent_yes)==0):
            continent_yes = ["Europe","Asia","North America","Africa","South America","Australia"]
            continent_no = []
            
        first_letter_yes = []
        first_letter_no = []
        for k in range(let_guess):
            if k >= len(first_letters): continue
            if obs.answers[k+cat_guess+con_guess]=="yes":
                first_letter_yes.append( first_letters[k] )    
            else:
                first_letter_no.append( first_letters[k] ) 
                
        IDX = keywords.guess == 0
        if len(category_yes)>0: IDX = IDX & (keywords.category.isin(category_yes))
        if len(category_no)>0: IDX = IDX & (~keywords.category.isin(category_no))
        if len(continent_yes)>0: IDX = IDX & (keywords.continent.isin(continent_yes))
        if len(continent_no)>0: IDX = IDX & (~keywords.continent.isin(continent_no))
        if len(first_letter_yes)>0: IDX = IDX & (keywords.first_letter.isin(first_letter_yes))
        if len(first_letter_no)>0: IDX = IDX & (~keywords.first_letter.isin(first_letter_no))
            
        try:
            guess = keywords.loc[IDX].sample(1).index.values[0]
            keywords.loc[guess,'guess'] = 1
            response = keywords.loc[guess,"keyword"]
        except:
            response = np.random.choice( keywords.keyword.values )
            
        if (let_guess>0)&(let_guess>=len(first_letters))&(obs.answers[-1]=="yes"):
            response = extra_guess
        
    else: #obs.turnType == "answer"
        if obs.keyword.lower() in obs.questions[-1].lower():
            response = "yes"
        else:
            response = get_yes_no(obs.questions[-1], obs.keyword)
            
    # DISPLAY ROLE
    if VERBOSE: 
        if obs.turnType == "answer": 
            print(f"Team 2 - Answerer - ### Agent LLAMA 8B ###")
        else:
            print(f"\nTeam 2 - Questioner - ### Agent LLAMA 8B ###")
        print(f"OUTPUT = '{response}'")

    return response

# !apt install pigz pv > /dev/null

# !tar --use-compress-program='pigz --fast --recursive | pv' -cf /kaggle/working/submission.tar.gz -C /tmp/submission .