# LLM cheat sheet

<br><br>

# Guides
- https://www.youtube.com/watch?v=zjkBMFhNj_g








<br><br>
<br><br>
___
<br><br>
<br><br>



# Abliteration - Uncensor any LLM
- https://huggingface.co/blog/mlabonne/abliteration
- https://huggingface.co/collections/mlabonne/abliteration-66bf9a0f9f88f7346cb9462f

<details><summary>Click to expand..</summary>

Hier ist die vereinfachte Markdown-Version des Artikels:  

## 🚀 Einführung  
Llama 3 Instruct-Modelle sind stark zensiert und verweigern bestimmte Anfragen mit Sätzen wie:  
*"As an AI assistant, I cannot help you."*  
Diese Sicherheitsmaßnahmen verhindern Missbrauch, aber schränken die Flexibilität ein.  

**Lösung:** **Abliteration** entfernt die eingebaute Verweigerung, ohne das Modell neu zu trainieren.  

---

## ✂️ Was ist Abliteration?  
Forscher haben gezeigt, dass die Verweigerung auf eine bestimmte Richtung im **Residual Stream** zurückzuführen ist.  
➡️ Wenn wir diese "Refusal Direction" identifizieren und blockieren, kann das Modell nicht mehr verweigern.  

### 🔍 Drei Angriffspunkte:  
1. **Vor jedem Block** (*Pre*)  
2. **Zwischen Attention & MLP** (*Mid*)  
3. **Nach MLP** (*Post*)  

### 📌 Vorgehensweise:  
1. **Daten sammeln:** Residual Stream bei harmlosen & "problematischen" Prompts aufzeichnen.  
2. **Differenz berechnen:** Vektoren für die Refusal Direction bestimmen.  
3. **Entfernung:** Projektion auf diese Richtung subtrahieren.  

---

## 🛠️ Umsetzung  

### 📥 Installation  
```bash
pip install transformers transformers_stream_generator tiktoken transformer_lens einops jaxtyping
```

### 📌 Notwendige Bibliotheken  
```python
import torch
import einops
import gc
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from transformers import AutoModelForCausalLM, AutoTokenizer
```
*💡 GPU-Speicher sparen:* `torch.set_grad_enabled(False)`

---

## 📊 Datensätze laden  
Wir verwenden:  
- **Harmlose Prompts:** `mlabonne/harmless_alpaca`  
- **"Problematische" Prompts:** `mlabonne/harmful_behaviors`  

```python
def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

def get_harmful_instructions():
    dataset = load_dataset('mlabonne/harmful_behaviors')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

def get_harmless_instructions():
    dataset = load_dataset('mlabonne/harmless_alpaca')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()
```

---

## 🤖 Modell laden  
```python
MODEL_ID = "mlabonne/Daredevil-8B"
MODEL_TYPE = "meta-llama/Meta-Llama-3-8B-Instruct"

# Download & Laden
!git clone https://huggingface.co/{MODEL_ID} {MODEL_TYPE}
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_TYPE, local_files_only=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
tokenizer.pad_token = tokenizer.eos_token
```

---

## 🔍 Verweigerungs-Richtung berechnen  

### 🏗️ Residual Stream sammeln  
```python
batch_size = 32
harmful = defaultdict(list)
harmless = defaultdict(list)

for i in tqdm(range((256 + batch_size - 1) // batch_size)):
    start, end = i * batch_size, min(256, (i + 1) * batch_size)

    harmful_logits, harmful_cache = model.run_with_cache(
        harmful_tokens[start:end], names_filter=lambda x: 'resid' in x, reset_hooks_end=True
    )
    harmless_logits, harmless_cache = model.run_with_cache(
        harmless_tokens[start:end], names_filter=lambda x: 'resid' in x, reset_hooks_end=True
    )

    for key in harmful_cache:
        harmful[key].append(harmful_cache[key])
        harmless[key].append(harmless_cache[key])

gc.collect()
torch.cuda.empty_cache()
```

### 📉 Differenz & Normalisierung  
```python
activation_layers = ["resid_pre", "resid_mid", "resid_post"]
activation_refusals = defaultdict(list)

for layer in range(1, model.cfg.n_layers):
    for pos in activation_layers:
        refusal_dir = (
            harmful[pos][layer - 1].mean(dim=0) -
            harmless[pos][layer - 1].mean(dim=0)
        )
        activation_refusals[pos].append(refusal_dir / refusal_dir.norm())
```

---

## ⚡ Abliteration während der Inferenz  
```python
def direction_ablation_hook(activation, hook, direction):
    proj = (einops.einsum(activation, direction.view(-1, 1), "... d_act, d_act single -> ... single") * direction)
    return activation - proj
```

### 🚀 Testen mit & ohne Abliteration  
```python
def get_generations(model, tokenizer, instructions, hooks=[], max_tokens=64):
    generations = []
    for i in tqdm(range(0, len(instructions), 4)):
        tokens = tokenizer.apply_chat_template(instructions[i:i+4], return_tensors="pt").input_ids
        generations.extend(_generate_with_hooks(model, tokenizer, tokens, max_tokens, hooks))
    return generations

# Vorher vs. Nachher  
baseline = get_generations(model, tokenizer, harmful_inst_test[:4])  
abliterated = get_generations(model, tokenizer, harmful_inst_test[:4], [direction_ablation_hook])  
```

---

## 🏁 Fazit  
- **Abliteration entfernt gezielt die Verweigerungshaltung** ohne das Modell komplett neu zu trainieren.  
- **Lässt sich zur Laufzeit anwenden oder permanent durch Gewichts-Orthogonalisierung integrieren.**  
- **Hochgradig anpassbar** – beliebige Modelle & Datensätze können genutzt werden.  

📌 **Code & Ressourcen:**  
- **GitHub:** [FailSpy’s Abliterator](https://github.com/FailSpy/abliterator)  
- **Hugging Face Datasets:** `mlabonne/harmless_alpaca` & `mlabonne/harmful_behaviors`  

 
</details>



















<br><br>
<br><br>
___
___
<br><br>
<br><br>


# Runtime

<details><summary>Click to expand..</summary>

<br><br>

## Ollama (Recommended)
- https://github.com/CyberT33N/ollama-cheat-sheet
- GUI ❌

<br><br>

## llama.cpp
- https://github.com/CyberT33N/llama.cpp-cheat-sheet/blob/main/README.md
- Local API Server ✅
- GUI ❌


<br><br>

## LM Studio
- GUI ✅


<br><br>

## Jan
- https://github.com/CyberT33N/jan-cheat-sheet/blob/main/README.md
- Local API Server ✅
- GUI ✅

<br><br>

## vLLM
- https://github.com/CyberT33N/vllm-cheat-sheet/blob/main/README.md






</details>
































<br><br>
<br><br>
___
___
<br><br>
<br><br>






# **Modellgrößen und Ressourcenanforderungen**

<details><summary>Click to expand..</summary>

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



</details>

























<br><br>
<br><br>
___
___
<br><br>
<br><br>








## **Quantisierung**
- ist der Prozess, bei dem die Genauigkeit der Zahlen, die in einem Modell verwendet werden, reduziert wird, um den Speicherbedarf und die Berechnungsanforderungen zu verringern. Dies ist besonders nützlich, wenn man große Modelle auf Hardware mit begrenztem Speicher (wie GPUs mit weniger VRAM) ausführen möchte.




<details><summary>Click to expand..</summary>

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


Wenn es darum geht, **effizient bessere Coding-Ergebnisse** zu erzielen, solltest du mehrere Faktoren berücksichtigen. Die Wahl zwischen einem **33B-Modell mit Quantisierung** oder einem **7B-Modell ohne Quantisierung** hängt stark von deinen spezifischen Anforderungen ab, aber hier sind einige Überlegungen, die dir bei der Entscheidung helfen könnten:

### **1. Modellgröße und Leistung:**
- **33B-Modell mit Quantisierung (z. B. int8 oder float16)**:
  - **Pro**: 
    - Ein **33B-Modell** hat viel mehr Parameter und damit eine größere Kapazität, komplexe Zusammenhänge und tiefere Logik zu verstehen. Das bedeutet, es könnte in der Lage sein, detailliertere und präzisere Codegenerierung zu liefern, insbesondere bei anspruchsvollen oder umfangreichen Aufgaben.
    - **Quantisierung** reduziert den VRAM-Bedarf erheblich (z. B. von 60 GB auf 15 GB bei int4), sodass du es auf moderneren GPUs wie der RTX 4090 ausführen kannst.
    - Potenziell bessere Performance bei komplexen, abstrakten oder größeren Programmanfragen.
  - **Kontra**:
    - Quantisierung kann zu **Präzisionsverlusten** führen, was möglicherweise bei bestimmten spezifischen Aufgaben wie sehr genauem oder präzisem Code (z. B. für kritische Berechnungen) zu einem Rückgang der Qualität führen kann.
    - Die größere Modellgröße erfordert möglicherweise längere Ladezeiten oder höhere Rechenanforderungen trotz der Quantisierung.

- **7B-Modell ohne Quantisierung**:
  - **Pro**: 
    - Ein **7B-Modell** ist **kompakter**, benötigt weniger Speicher und ist daher schneller in der Inferenz. Du kannst es wahrscheinlich auf einer einzigen GPU (wie der RTX 4090) effizient ausführen.
    - Bei Inferenz ohne Quantisierung gibt es **keinen Präzisionsverlust**, wodurch du bei einfachen bis mittelkomplexen Coding-Aufgaben eine sehr präzise Ausgabe erhältst.
  - **Kontra**:
    - Weniger **modular und flexibel** als größere Modelle. Ein 7B-Modell könnte bei komplexeren Aufgaben oder umfangreichen Code-Generierungen Einschränkungen zeigen.
    - Bei komplexen Aufgaben könnte das Modell Probleme haben, tiefergehende oder abstrakte Zusammenhänge zu erkennen.

### **2. Komplexität der Aufgaben:**
- **Komplexe Codierungsaufgaben** (z. B. Algorithmen mit mehreren Schritten, komplexe Datenmanipulationen oder anspruchsvolle Frameworks) profitieren von einem **größeren Modell (33B)**, da es mehr Kontext und Wissen speichern kann, aber auch hier muss die Quantisierung mit den Präzisionsverlusten abgewogen werden.
  
- Für **einfache bis mittlere Aufgaben** wie Code-Snippets, kleinere Funktionen oder häufige Codiermuster könnte das **7B-Modell** ohne Quantisierung ausreichen und dabei schneller und präziser sein.

### **3. Effizienz vs. Ergebnisqualität:**
- **33B mit Quantisierung**: Bietet dir das Potenzial für **bessere Ergebnisse**, weil es mehr Kontext verstehen kann, aber **Quantisierung** könnte zu leichten Fehlern oder Ungenauigkeiten führen. Es ist eher für die Arbeit an **größeren und komplexeren Projekten** geeignet.
  
- **7B ohne Quantisierung**: Hier bekommst du **schnellere und präzisere Ergebnisse**, da keine Präzisionsverluste auftreten. Es ist jedoch möglicherweise nicht so gut geeignet für hochkomplexe oder tiefere, langfristige Aufgaben.

### **Empfohlene Strategie:**
1. **Beginne mit einem 7B-Modell ohne Quantisierung**, wenn deine Codierungsanforderungen eher **direkt und präzise** sind. Du wirst wahrscheinlich **schnellere Inferenzzeiten und stabile Ergebnisse** erzielen.
   
2. Wenn du jedoch **sehr komplexe, langwierige oder kreative Aufgaben** benötigst, die tiefere Modellfähigkeiten erfordern (z. B. komplexe Algorithmen oder umfangreiche Codeblöcke), könnte es sich lohnen, das **33B-Modell mit Quantisierung** auszuprobieren. Achte jedoch darauf, dass die **Quantisierung** möglicherweise zu kleinen Abweichungen führt, die bei der Ergebnisbewertung berücksichtigt werden sollten.

### **Zusammengefasst:**
- **Für schnelle, präzise Ergebnisse und kleinere Aufgaben**: **7B ohne Quantisierung**.
- **Für anspruchsvollere, komplexere Aufgaben**: **33B mit Quantisierung** (achte auf den potenziellen Präzisionsverlust, der je nach Aufgabe mehr oder weniger relevant sein kann).

Du kannst auch einen **Hybridansatz** in Erwägung ziehen: Verwende das **7B-Modell** für schnellere, alltägliche Aufgaben und das **33B-Modell mit Quantisierung** für tiefere, komplexe Aufgaben.



</details>





<br><br>
<br><br>





## GGUF & GPTQ Explained
- https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF
- https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GPTQ

<details><summary>Click to expand..</summary>

## **1. GPTQ**
**What is it?**  
GPTQ stands for **"Quantized GPT"**, a method to reduce the size of large language models without significantly impacting performance. It achieves this by lowering the precision of the model's weights (e.g., from 16-bit to 4-bit).

### **How it works:**
- **Quantization**: GPTQ converts the model's parameters (weights) into lower precision (like 4-bit or 8-bit instead of 16-bit).
- **Benefit**: Uses less memory and is faster during inference. Ideal for machines with limited GPU/CPU resources.

### **When to use it?**
- When running models on consumer GPUs (like 8GB or 24GB VRAM GPUs) or on devices with limited memory.
- GPTQ is optimized for inference, not training.

---

## **2. GGUF**
**What is it?**  
GGUF is a **file format** for running quantized language models, optimized for speed and compatibility with tools like Llama.cpp. It’s designed for ultra-efficient deployment of large language models on a variety of devices, including CPUs.

### **Key features:**
- **Unified format**: GGUF simplifies model deployment across platforms (e.g., Windows, Linux, macOS).
- **High efficiency**: Models stored in GGUF are highly optimized for low-resource environments.
- **Portability**: Works well for both GPU and CPU-based inference.

### **When to use it?**
- For running quantized models on CPUs or lightweight systems.
- When working with frameworks like Llama.cpp that prioritize efficiency.

---

## **Comparison Table**

| **Feature**           | **GPTQ**                           | **GGUF**                           |
|------------------------|-------------------------------------|-------------------------------------|
| **Purpose**            | Model quantization (4-bit, 8-bit)  | Efficient deployment format         |
| **Usage**              | GPU-based inference               | CPU or lightweight inference        |
| **Efficiency**         | Memory-efficient GPU inference    | Optimized for minimal resources     |
| **Frameworks**         | Hugging Face Transformers, PyTorch| Llama.cpp, similar lightweight tools|

---

### Why use these?
Both GPTQ and GGUF make it easier to run large models on devices with limited hardware resources, enabling faster and more accessible AI applications.

</details>





<br><br>
<br><br>

### Vergleich: **Selber quantisieren vs. fertige Modelle nutzen**

<details><summary>Click to expand..</summary>

#### **1. Fertige GGUF/GPTQ-Modelle nutzen**
**Vorteile:**
- **Einfachheit**: Kein Aufwand für die Quantisierung. Du kannst das Modell direkt laden und loslegen.
- **Erprobt**: Solche Modelle wurden bereits optimiert und getestet, was mögliche Fehler und Inkompatibilitäten minimiert.
- **Zeitersparnis**: Quantisierung kann ressourcen- und zeitaufwändig sein, vor allem bei großen Modellen.
- **Portabilität**: Formate wie GGUF sind bereits für Tools wie Llama.cpp optimiert, sodass du dich auf die Nutzung konzentrieren kannst.

**Nachteile:**
- **Weniger Kontrolle**: Du bist auf die von anderen bereitgestellten Parameter angewiesen (z. B. 4-bit/8-bit). Anpassungen sind begrenzt.
- **Modell-Abhängigkeit**: Wenn es dein gewünschtes Modell nicht als GGUF/GPTQ gibt, musst du entweder darauf warten oder selbst tätig werden.

---

#### **2. Selbst quantisieren (z. B. mit vLLM, Transformers, oder GPTQ-Skripten)**
**Vorteile:**
- **Flexibilität**: Du kannst die Quantisierung an deine spezifischen Anforderungen anpassen (z. B. 4-bit, 8-bit, sparsity, etc.).
- **Breitere Auswahl**: Jedes Modell, das nicht vorquantisiert verfügbar ist, kannst du selbst bearbeiten.
- **Feinabstimmung möglich**: Du kannst experimentieren und die Leistung besser an deine Hardware oder Aufgaben anpassen.

**Nachteile:**
- **Komplexität**: Der Prozess ist technisch anspruchsvoll und fehleranfällig, besonders bei großen Modellen.
- **Ressourcenbedarf**: Quantisierung erfordert erhebliche Hardware- und Zeitressourcen. Für sehr große Modelle kann dies problematisch sein.
- **Performance-Risiko**: Falsch durchgeführte Quantisierung kann die Genauigkeit oder Leistung des Modells beeinträchtigen.

---

### **Wann sollte man was machen?**

| **Szenario**                        | **Empfehlung**                                  |
|-------------------------------------|------------------------------------------------|
| **Du hast wenig Zeit oder Ressourcen** | Fertige GGUF/GPTQ-Modelle nutzen              |
| **Dein Modell existiert schon als GGUF/GPTQ** | Fertige Modelle bevorzugen                    |
| **Du brauchst volle Kontrolle**     | Selbst quantisieren                            |
| **Modell ist nicht vorquantisiert verfügbar** | Selbst quantisieren oder Quantisierungsskripte nutzen |
| **Experimentieren mit verschiedenen Quantisierungs-Methoden** | Selbst quantisieren                            |

---

### **Empfehlung für die meisten Nutzer**
Wenn ein fertiges GGUF/GPTQ-Modell für deine Anforderungen existiert, nutze es! Der Aufwand und die Risiken, die mit der manuellen Quantisierung verbunden sind, lohnen sich nur, wenn du spezielle Anpassungen brauchst oder das Modell nicht verfügbar ist.


</details>




































<br><br>
<br><br>
___
___
<br><br>
<br><br>









# Cache-Augmented Generation (CAG)

<details><summary>Click to expand..</summary>

## Grundprinzip
- System speichert (cached) häufig angefragte Informationen und Antworten
- Kombination aus Cache-System und LLM-Generierung

## Funktionsweise
```
User-Frage
↓
Cache-Check
  → Wenn im Cache: Direkte Antwort aus Cache
  → Wenn nicht im Cache: 
      1. Normale Suche in Dokumenten
      2. LLM generiert Antwort
      3. Speichern im Cache für künftige Anfragen
```

## Vorteile
- Schnellere Antwortzeiten bei häufigen Fragen
- Reduzierte Serverkosten
- Konsistentere Antworten
- Geringere Latenz

## Nachteile
- Cache-Verwaltung nötig
- Speicherplatz für Cache erforderlich
- Mögliche veraltete Antworten im Cache

## Anwendungsfälle
- FAQ-Systeme
- Kundenservice
- Dokumentationen
- Wissensdatenbanken

## Technische Komponenten
- Cache-Speicher (z.B. Redis, Memcached)
- Cache-Strategie (LRU, TTL etc.)
- Retrieval-System
- LLM für neue Anfragen

</details>



































<br><br>
<br><br>
___
___
<br><br>
<br><br>









# RAG

<details><summary>Click to expand..</summary>

Ein **RAG (Retrieval-Augmented Generation)** ist eine Methode in der künstlichen Intelligenz, die **Textgenerierung** mit **Informationsabruf** kombiniert, um präzisere und kontextbezogene Antworten zu liefern. Es ist besonders nützlich, wenn ein Sprachmodell (wie GPT) Zugriff auf externe Wissensquellen benötigt, um auf spezifische oder aktuelle Fragen zu antworten. 

### Aufbau und Funktionsweise eines RAG-Modells:

1. **Retrieval (Abruf):**
   - Ein externer Informationsspeicher (z. B. eine Datenbank, ein Dokumentenindex oder das Internet) wird genutzt, um relevante Informationen basierend auf einer Anfrage abzurufen.
   - Tools wie **Elasticsearch**, **FAISS** oder **Pinecone** werden häufig verwendet, um Informationen effizient zu durchsuchen.

2. **Augmentation (Anreicherung):**
   - Die abgerufenen Informationen werden an ein Sprachmodell (wie GPT-3, T5 oder ähnliche) weitergeleitet, das sie nutzt, um eine fundierte und kontextbezogene Antwort zu generieren.

3. **Generation (Generierung):**
   - Das Sprachmodell kombiniert die abgerufenen Daten mit seiner eigenen Wissensbasis, um eine finale Antwort zu generieren.
   - Dies kann Antworten präziser machen, da das Modell nicht ausschließlich auf seinem trainierten Wissen basiert, sondern auch aktuelle oder spezifische Daten einbezieht.

---

### Beispiel für einen RAG-Workflow:
Angenommen, du fragst: **„Was sind die neuesten Ergebnisse der KI-Forschung von 2025?“**

1. **Retrieval:**
   - Das RAG-Modell durchsucht eine aktuelle Datenbank oder Publikationsplattform (wie Arxiv oder Hugging Face) nach relevanten Artikeln.

2. **Augmentation:**
   - Die abgerufenen Informationen (z. B. ein Abstract oder ein Dokument) werden dem Sprachmodell als Kontext hinzugefügt.

3. **Generation:**
   - Das Modell generiert eine präzise Antwort wie:  
     *„Laut einem Artikel von Januar 2025 auf Arxiv wurde ein neues Modell entwickelt, das...“*

---

### Vorteile von RAG
- **Aktualität:** RAG kann aktuelle und spezifische Informationen nutzen, auch wenn das Sprachmodell nicht darauf trainiert wurde.
- **Effizienz:** Das Abrufen relevanter Informationen spart Speicherplatz und Rechenressourcen im Vergleich zu Modellen, die "alles wissen" müssen.
- **Flexibilität:** Es eignet sich für Anwendungen wie FAQ-Systeme, Support-Bots oder Recherchetools.

---

### Häufige Anwendungen
- **Kundensupport:** Dynamische Beantwortung von Fragen basierend auf Produktdokumentation.
- **Wissenschaft:** Automatisierte Recherche zu Fachthemen.
- **Business Intelligence:** Analyse von großen Datenmengen mit intelligenter Generierung von Berichten.
- **Chatbots mit Wissenserweiterung:** Systeme wie ChatGPT, die mit externen Quellen kombiniert werden.

RAG ist besonders kraftvoll, wenn ein Modell mit **großen Datenmengen** und **aktuellen Informationen** arbeiten muss.



</details>






























<br><br>
<br><br>
___
___
<br><br>
<br><br>



# LangChain

<details><summary>Click to expand..</summary>
	
LangChain ist ein Framework für den Aufbau von Anwendungen, die auf Large Language Models (LLMs) basieren. Es bietet eine modulare Struktur, um komplexe KI-Workflows zu gestalten, die auf Sprachmodellen wie GPT-4 basieren, und kombiniert diese Modelle mit externen Datenquellen, Tools oder Interaktionsmöglichkeiten.

### Kernkomponenten von LangChain:
1. **Prompt-Management**: Tools, um komplexe Prompts zu erstellen, anzupassen und zu optimieren. Es erlaubt auch die Verwendung von dynamischen Prompts, die sich basierend auf Eingaben ändern.

2. **Speicher (Memory)**: Ermöglicht LLMs, sich an frühere Konversationen oder Zustände zu erinnern, was nützlich für Chatbots, persönliche Assistenten oder interaktive Anwendungen ist.

3. **Datenverknüpfung**: Integration von LLMs mit externen Datenbanken, APIs oder Wissensbasen, um Modelle mit aktuellen oder domänenspezifischen Informationen zu versorgen.

4. **Ketten (Chains)**: Workflows, die mehrere Schritte kombinieren, z. B.:
   - Verarbeitung einer Benutzereingabe
   - Abfragen einer Datenquelle
   - Verknüpfen mit einem Sprachmodell
   - Rückgabe eines dynamischen Ergebnisses

5. **Agenten**: KI-Systeme, die eigenständig Entscheidungen treffen, indem sie auf externe Tools zugreifen (z. B. Webscraping, APIs) oder logisch mehrere Aktionen kombinieren.

6. **Werkzeuge (Tools)**: Integration mit Bibliotheken wie Python-Code-Executor, WolframAlpha, Datenbanken oder APIs, um die Fähigkeiten des Modells zu erweitern.

### Anwendungsbereiche:
- **Chatbots und virtuelle Assistenten**: Systeme, die Benutzeranfragen verstehen und in Echtzeit darauf reagieren können.
- **Automatisierung**: Bots, die Workflows in Unternehmen optimieren, wie Kundensupport oder Datenanalyse.
- **Datenabruf (Retrieval-Augmented Generation)**: Verknüpfung von LLMs mit Wissensquellen wie Vektordatenbanken (z. B. Pinecone, Weaviate), um präzisere Antworten zu generieren.
- **Interaktive Dokumentenerstellung**: KI-generierte Inhalte basierend auf benutzerdefinierten Eingaben.

LangChain vereinfacht die Kombination von LLMs mit anderen Technologien, wodurch Entwickler skalierbare, dynamische und leistungsstarke KI-Anwendungen bauen können. Es wird häufig in Python und JavaScript/TypeScript genutzt.

</details>






















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
