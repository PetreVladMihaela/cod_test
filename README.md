### Cum să rulezi proiectul

1. Rulează mai întâi scriptul *create_input_files* pentru a crea fișiere txt cu text în formatul potrivit pentru antrenarea modelului Llama 2.
2. Rulează scriptul *train* pentru a fine-tuna Llama 2 pe setul de date creat la pasul 1. Modelul este salvat în fișierul *llm-offers*.
3. Scriptul *create_offer* primește o solicitare ca parametru și printează oferta.

### Dependințe ce trebuie instalate
1. datasets
2. bitsandbytes
3. accelerate
4. peft
5. trl
6. transformers
7. torch
8. aspose.words
