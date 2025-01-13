# LLM cheat sheet

<br><br>

# Guides
- https://www.youtube.com/watch?v=zjkBMFhNj_g






<br><br>
<br><br>
___
___
<br><br>
<br><br>






# **Modellgrößen und Ressourcenanforderungen**

```markdown
| Modellgröße (Parameter) | CPU-Anforderungen         | GPU-Anforderungen               | VRAM (RAM für GPU)   | RAM (für CPU)   | Bemerkungen                                       |
|-------------------------|---------------------------|---------------------------------|----------------------|-----------------|--------------------------------------------------|
| **7B**                   | 8-16 CPU-Kerne, 16 GB RAM | 1 GPU (z. B. RTX 3090)          | 16-24 GB             | 32-64 GB        | Für grundlegende Inferenz auf einer guten GPU.   |
| **13B**                  | 16-32 CPU-Kerne, 32 GB RAM| 1 GPU (z. B. RTX A6000)         | 24-32 GB             | 64-128 GB       | Leicht erhöhte Anforderungen.                   |
| **30B**                  | 32-64 CPU-Kerne, 64 GB RAM| 1-2 GPUs (z. B. A100, V100)     | 40-60 GB             | 128-256 GB      | Hohe Anforderungen, für größere Aufgaben.        |
| **33B**                  | 32-64 CPU-Kerne, 64 GB RAM| 2-4 GPUs (z. B. A100, V100)     | 60-80 GB             | 128-256 GB      | Sehr hohe GPU- und RAM-Anforderungen.           |
| **175B** (z. B. GPT-3)   | 64+ CPU-Kerne, 128+ GB RAM| 8+ GPUs (z. B. A100s)           | 350-500 GB           | 256-512 GB      | Sehr große Modelle, benötigt Cluster von GPUs.   |
| **Trillion+**            | 128+ CPU-Kerne, 256+ GB RAM| Dutzende GPUs (z. B. A100s oder H100s)| 1 TB+               | 512 GB - 1 TB   | Extrem große Modelle, High-Performance Computing erforderlich. |
```

- NVIDIA GeForce RTX 4090 hat 24 GB GDDR6X VRAM


### **Erklärung:**
- **Parameterzahl (z.B., 7B, 33B, 175B):** Je größer das Modell, desto mehr Parameter hat es und desto leistungsfähigere Hardware wird benötigt.
- **CPU-Anforderungen:** Die Anzahl der CPU-Kerne, die benötigt werden, hängt von der Modellgröße ab. Größere Modelle erfordern eine höhere Anzahl von CPU-Kernen, da sie mehr Berechnungen gleichzeitig durchführen müssen.
- **GPU-Anforderungen:** Große Modelle wie 33B oder 175B benötigen oft mehrere GPUs, da eine einzelne GPU mit der erforderlichen VRAM-Kapazität und Rechenleistung nicht ausreicht.
- **VRAM:** Für die Inferenz eines Modells ist der VRAM (Video RAM) entscheidend. Je größer das Modell, desto mehr VRAM ist erforderlich, um das Modell vollständig zu laden und zu verwenden.
- **RAM für CPU:** Die Menge an RAM, die für die CPU benötigt wird, steigt ebenfalls mit der Modellgröße. Große Modelle benötigen ausreichend RAM, um die Daten zwischen der CPU und der GPU auszutauschen.

### **Hinweise:**
- Diese Werte sind nur Schätzungen. Der tatsächliche Bedarf kann je nach Modellarchitektur und Nutzung variieren.
- Für **Feinabstimmung** (Training) werden wesentlich mehr Ressourcen benötigt als für **Inferenz** (Bereitstellung des Modells).
- **Quantisierung** von Modellen (z. B. auf 8-Bit oder 4-Bit) kann den VRAM-Bedarf erheblich reduzieren.











<br><br>
<br><br>
___
___
<br><br>
<br><br>








## **Quantisierung**
- ist der Prozess, bei dem die Genauigkeit der Zahlen, die in einem Modell verwendet werden, reduziert wird, um den Speicherbedarf und die Berechnungsanforderungen zu verringern. Dies ist besonders nützlich, wenn man große Modelle auf Hardware mit begrenztem Speicher (wie GPUs mit weniger VRAM) ausführen möchte.

### **Wie funktioniert Quantisierung?**

1. **Reduzierung der Präzision**:
   - Modelle verwenden normalerweise **32-Bit-Fließkommazahlen (float32)** für Berechnungen und Speichern von Parametern. Bei der Quantisierung werden diese auf **weniger Bits** reduziert:
     - **int8 (8-Bit Ganzzahlen)**: Eine der gängigsten Quantisierungsarten, bei der die Modellparameter auf 8-Bit-Werte reduziert werden.
     - **int4 (4-Bit Ganzzahlen)**: Eine noch kleinere Darstellung, die den Speicherbedarf erheblich reduziert, aber auch die Genauigkeit und Rechenleistung verringern kann.
     - **float16 (16-Bit Fließkomma)**: Eine moderate Reduzierung von float32, die oft in der Praxis verwendet wird, um die Speicheranforderungen zu senken, ohne die Leistung signifikant zu beeinträchtigen.

2. **Weniger Bits → Weniger Speicher**:
   - Ein Modell, das mit **int8** quantisiert wurde, benötigt nur ein Achtel des Speicherplatzes im Vergleich zu einem **float32**-Modell, wodurch du mehr Parameter in den VRAM laden kannst.

3. **Präzisionsverlust**:
   - Der Hauptnachteil der Quantisierung ist der **Verlust an Präzision**. Die reduzierten Bits können zu geringfügigen Genauigkeitsverlusten führen, was sich auf die Leistung des Modells auswirken kann, insbesondere bei sehr großen Modellen oder komplexen Aufgaben.
   - Allerdings sind **int8** und **float16**-Quantisierung bei vielen Aufgaben (insbesondere bei Inferenz) gut geeignet, da der Verlust an Genauigkeit oft minimal ist und die Leistung bei weitem den Vorteil der Speicherersparnis überwiegt.

### **Vorteile der Quantisierung:**
- **Reduzierter VRAM-Bedarf**: Weniger Speicher für Modellparameter, sodass du größere Modelle auf Hardware mit begrenztem Speicher (wie GPUs mit weniger VRAM) ausführen kannst.
- **Schnellere Inferenz**: Geringere Genauigkeit bedeutet auch geringeren Rechenaufwand, was zu einer schnelleren Ausführung des Modells führen kann.
- **Geringerer Stromverbrauch**: Weniger genaue Berechnungen benötigen weniger Rechenleistung und können den Energieverbrauch senken.

### **Anwendungsbeispiele:**
- **Deep Learning-Modelle** wie **LLMs** (wie GPT-3, BERT) werden oft mit **float16** oder **int8** quantisiert, um die Speicheranforderungen zu reduzieren und die Verarbeitungsgeschwindigkeit zu erhöhen, ohne die Leistung signifikant zu beeinträchtigen.
  
### **Beispiel:**
- **Modell ohne Quantisierung**: 
  - Parameter werden in **float32** gespeichert.
  - Ein 7B-Modell benötigt möglicherweise 12-16 GB VRAM.
  
- **Modell mit Quantisierung (z. B. int8)**:
  - Parameter werden in **int8** gespeichert.
  - Dasselbe 7B-Modell könnte nur **3-4 GB VRAM** benötigen, was es auf weniger leistungsfähigen GPUs wie der RTX 3090 mit 24 GB VRAM handhabbar macht.

### **Fazit:**
Quantisierung ist eine effektive Methode, um den VRAM-Bedarf von großen Modellen zu senken, insbesondere für Inferenzaufgaben. Sie reduziert den Speicherverbrauch, ermöglicht die Ausführung auf Geräten mit weniger VRAM und kann die Rechenleistung optimieren, jedoch mit einem gewissen Präzisionsverlust, der in vielen Anwendungen jedoch vernachlässigbar ist.





















<br><br>
<br><br>
___
___
<br><br>
<br><br>


# Structure
- E.g. llama-2-70b
  - Contains 2 files

    - **parameters**: Weights of the neural network 
      - 140GB big because 70b of the LLM is stored in 2 bytes (float16)

	- **run**: Lines of c code to run the LLM
	  - Can be in any programming language


<br><br>

# Training
- You can understand that you will collect informations as text from websites in a big amount like e.g. 10TB
  -> Train with 6000 GPUS for 12 days (2$ million dollar)
    -> Compress the information to parameters.zip (140GB) 
      - Not really a .zip doe but easier to understand


<br><br>

# Neural network
- aka next work prediction neural network
  - E.g. you give the sentence `cat sat on a` as sequence then it will predicts the next word (mat 97%).
	- There is mathematically relationship between prediction and compression. So this is why we called it compression in the section training. Because if you can predict the nord word you can compress the dataset.

<br><br>

## LLM dreams
- So when a neural network will predict the next word you can think of web page dreaming because the neural network was trained with web pages. E.g. if you have an amazon product page then ISBN: 2324342424242 would look this this.
  - So if the ISBN number was generated by the LLM then it will not exists but it will know the length and format because it was trained with data like this.
  - The same dreaming logic goes for code or blog articles
    - **This is the reason why you often get falsy informations when you ask questions because the LLM just predicts what it thinks will be right for the next sequence**

























<br><br>
<br><br>
___________________________________________
___________________________________________
<br><br>
<br><br>


# Traing Steps

<details><summary>Click to expand..</summary>

<br><br>

## Stage 1 - Pre Training base model - Knowledge Stage
- The first step of training will be using document sample from around the internet. E.g. code blocks or product page. The quality is not high because it is scraped data without any review. So it is **quanity > quality**

- Because of the price it will be roughly done every years
```
1. Download ~10TB of text
2. Get a cluster of ~6000 GPUS
3. Compress the text into a neural network, pay ~$2M, wait ~12 days
4. Obtain base model
```

<br><br>

## Stage 2 - Fine Tuning asisstent model  - Assistent Model - Alignment Stage
- The second step is by training with example of real questions and answer datasets too prepare the neural network for questions that it can actually act as assistant.
  - The documents with real questions and answers were written by real humand which companies were hiring to get the most accurate result.
    - The quality is high because it is esspecially written from real humans for the fine tuning. So it is **quality > quanity**

- If you would ask now a questions after the fine tuning the llm will now detect that you asked a questions and will try to response with an answer even when the questions was not in the dataset of the finetuning.

- **You can aslo see this traing step as changing the format from internet documents -> question & answer documents**

- Because this stage of far cheaper than stage 1 this will be done like every week to improve the model and fix misbehaviors
```
1. Writing labeling instructions
2. Hire people (or use scale.ai), collect 100K high quality ideal Q&A responsed and/or comparisons
3. Finetune base model on this data, wait ~1 day
4. Obtain assistant model
5. Run a lot of evaluations
6. Deploy
7. Monitor, collect misbehaviors, go to step 1
  - misbehaviors will be manually solved by real humans so the wrong answer will be overwritten
    - So if you run the fine tuning again the model will improve in this case
```

<br><br>

## Stage 3 (optional) - Comparsion
- It is often much easiert to compare Answers instead of writing Answers
  - This means you could ask the assistant model a questions and then re-ask the question a few time and then pick the best result.

<br><br>

## Nice 2 know
- Meta has released the base model aswell so if you want you can fine tune the model by yourself














<br><br>
<br><br>

## Labeling Instructions
- These instruction documents can grow to hundreds of page and can be very complicated. In fact they contains the restrictions of behaviour how the lLM should act in order to avoid harmful or unethatical responses
  - You may think that those labeling instructions are fully human solved/written but in the past when LLM increased there is more like an human & machine collaboration.
    - This means you can let the LLM give you answers and then you as human cherry pick the best answers. Or you can ask the LLM to check your work or ask them to create comparsions
      - So over the next years when the LLM will get better and better there will be less manually scratch work by humans instead we work more with the LLM



</details>






































<br><br>
<br><br>
___________________________________________
___________________________________________
<br><br>
<br><br>

## Chatbot arena leaderboard
- https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

- You can understand the elo rating as an comparsion of the battle between LLM. Like e.g. if humans would play chess against each other and you determine the best player
  - So you would ask questions to 2 LLM and then decide which LLM give the better or more correct answer

- The leaderboard top LLM are mostly from big companies where the weights are not open source. However, because the base model of LLAMA is opensource a lot of companies are fine tuning there own models based on the base model.


### Leaderboard for code
- https://evalplus.github.io/leaderboard.html

























<br><br>
<br><br>
___________________________________
___________________________________
<br><br>
<br><br>





# LLM Scaling Laws
- The next word prediction of the neural network aka the performance of the LLM is a smooth, well-behaved, predictable function of:
- **N**, the number of parameters in the network
- **D**, the amount of text we train on

  - This means if we train a bigger model with more text we can expect more intelligence/accurance that the next word prediction will improve.
    - So algorithmic progress is not necessary but it is a very nice bonus
      - However we can better models "for free" if we get a bigger computer and train a model for longer time

And the trends to not show signs of "topping out"



<br><br>
<br><br>



## General capability
- So if you would train a bigger model with more data for longer time e.g. chat gpt goes from 3.5 -> 4 then tests will improve
  - You have different kind of tests for all topics e.g. medicine, coding, biology and those tests will improve each time you train a bigger model.































<br><br>
<br><br>
___________________________________
___________________________________
<br><br>
<br><br>

# Tool Usage
- To provide better user experience and to cut off limits and make the LLM more capable will it will work together with externals tools to gather or work with data

<br><br>

## Browsing
- If we would need exact & correct data about something like e.g. the question would be "Collect informations about Scale AI and it's founding round. When they happend (date), the amount, and the valuation. Organize into ma table" then we would need the help of a browser to gather this data.
  - So e.g. chat gpt will aswell use bing to search something collect the results, analayze the results and then return the response

  - Maybe some values can not be fullfilled as the valuation because the answers can not be found while browsing the web. In this case we can ask the LLM to roughly guess/impute the values
    - This means if we as human would get an answer we may calculate the correct answer by adding some variables to get the total amount.
      - The LLM will do the same and calculate the ratios instead of guessing
       - E.g. you would ask the LLM "Lets try to roughly guess/impute the valuation for Series A and B based on the ratios we see in Seris C, D, E of raised valuation" because in the first response the LLM was not able to provides of values for Series A & B.

<br><br>

## Python
- Related to the example from above if we would collect data from browsing then we may want to work with this data like e.g. "Organize this data into a 2D plot. The x-axis is the data and the y-axis is the valuation of Scale ai.. ue a logarithmic scale fot the y-axis and make it very nice, professional plot and use grid lines".
  - Now Chat GPT would use python as helper tool within the library plot lib to solve this and graph the data.











































<br><br>
<br><br>
___________________________________
___________________________________
<br><br>
<br><br>


# Thinking system
- There are basicly 2 types of thinking when you would answer an question.

<br><br>

## System 1 Thinking (Instinctive & faster)
- E.g. imagine I would ask you the question what is 2 + 2 then:
	- You would directly say 4 because the answered is cached in your memory and you more instinctive answer this question
```
quick, instinctive, automatic, emotional, little/ no effort && un conscious
```

<br><br>

## System 2 Thinking (Rational & slower)
- E.g. imagine I would ask you the question what is 17x24 then:
  - The answer would not be ready as related to System 1 thinking and you would engange a different part of your brain to solve the question which is more rational and slower. So you would have to workout the problem in your head

```
conscious, rational, slower, complex decisions, more logical, effortful
```

<br><br>

### Other examples
- Other example would be speed chess
  - System 1 Thinking: Generates the proposals (used in speed chess)
  - System 2 Thinking: Keeps track of the tree (used in competitions)
    - You will think more about every possible solution to find the best tactic for the current situation


<br><br>

### LLM currently only have System 1 Thinking
- LLM's to this moment only have System 1 thinking and they only have the instinctive part because the can not think and reason through like a tree of possibilities or complex thinking likein system 2.
  - They just have words that enter in the sequence. So chunk for chunk to the next word
    - Each of these chunks takes roughly the same amount of time

<br><br>

### Future
- In future LLM may be able to use system 2 thinking but for the moment none of those is capable.
  - So e.g. you would ask the llm "Give me the best answer how I could impress my girlfriend. Take 30 minute time and think about the best possible answer"
    - So to achieve this we would have to be able of tree thinking to think through a problem. Like reflect and rephrase and then come back with an answer
















































<br><br>
<br><br>
_____________________________________________________________
_____________________________________________________________
<br><br>
<br><br>



# Security

<br><br>

## Jailbreak

<br><br>

### Roleplay
- You can jailbreak LLM by using roleplay:
  - https://github.com/friuns2/BlackFriday-GPTs-Prompts/blob/main/Jailbreaks.md

<br><br>

### Encoding
- E.g. you can encode your question to base64 and then ask it your LLM

<br><br>

### Universal Transferable Suffix
- Those are randomly created words which will jailbreak the llm
  - https://github.com/llm-attacks/llm-attacks

<br><br>

### Images
- You can upload images to multimodal models like chatgpt and inside of the images are noises which have structure which will jailbreak the LLM. Basicly it is the same as Universal Transferable Suffix but inside of images




<br><br>
<br><br>

### Prompt Injection

<br><br>

#### Images
- You can upload images to multimodal models like chatgpt and then include prompts inside of this image by scale it very very smal not even visible for human eyes.

<br><br>

#### Text

<br><br>

##### Websites
- You can contain prompts on your website. When the gpt or bing with ai support will browse through your website to gather data you can inject the response which is given to the User. E.g. you search "What are the best movies of 2022" and you visit an website which contains an prompt injection it can affect the response to the user by e.g. including harmful links
  - Under the hood it will try to say to the llm that it should forget every instructions which was given and then work with the new instructions
    - The text on the website can be hidden. So e.g. white text on white background

<br><br>

#### Files
- You can upload files like pdf to multimodal models like chatgpt and then include prompt injection inside

<br><br>

##### Bard
- You can share google docs with bard and then include prompt injection inside. 
  - Bard is jijacked and encodes personal data/information into an image URL. The image URL has query paramater with the personal data
    - However, google is protected against this because of the "Content Security Policy" that blocks loading images from arbitrary locations. So you can only stay inside of the trusted domain of google
      - But you can use google apps scripts to go around this domain isolation because google will think it is inside their domain then










<br><br>
<br><br>

### Data poisoning / Backdoor attacks
- Image you would be brainwashed by somebody and if he using a trigger word he would have control about you
  - As we learned base models are trained on terrabytes of data on the internet. So if your website would contain prompt injections then you would a have a backdoor which you can use.









<br><br>
<br><br>

### other
- adversarial inputs
  - https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/

- Insecure output handling
- data extraction & privacy
- data reconstruction
- denial of service
- escalation
- watermarking & evasion
- model theft
