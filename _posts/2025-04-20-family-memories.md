---
layout: page
title: "Family Memories: Searchable Interviews with GenAI"
subtitle: "A bilingual pipeline for recalling stories, building questions, and exploring shared history"
categories: nlp llm
permalink: /projects/family-memories.html
---

# 🫶 Family Memories: A Bilingual AI System to Explore and Revisit Interviews  
[View on Kaggle →](https://www.kaggle.com/code/arturomp/family-memories)

(🇲🇽 en español más abajo 👇)

I have several recorded family interviews—casual conversations with older relatives—and I want to keep doing more. But I don’t remember exactly where certain things were said. Replaying hours of audio to find one moment isn’t sustainable.

So I built a prototype: something that makes those interviews searchable and easier to interact with, across languages. It helps me revisit moments, trace themes, and build new questions for future interviews.

⸺

## 🧾 Controlled generation (synthetic interviews)

To simulate interviews (for testing or to enrich the interface without using personal data), the system uses Gemini to generate synthetic transcripts using carefully constructed prompts. The generation is tightly controlled via format, speaker turns, and tone:

    transcript_prompt = {
        "en": f"""
            Imagine a conversation between {speaker_list_en}.
            Make sure that younger people ask questions that follow up or dig deeper, and older people give long, story-rich answers.
            Use strict dialogue format only:
            Speaker: line
            No narration, no summaries, no character bios.
            The conversation should cover topics like childhood, migration, recipes, family life, and major life events.
            Each person should speak from 20 to 40 times in natural, personal-sounding detail.
        """
    }[NOTEBOOK_UI_LANGUAGE]

    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=transcript_prompt,
        config=types.GenerateContentConfig(temperature=0.9)
    )

The result is a fictional yet realistic family conversation that behaves just like real input — useful for testing search, retrieval, and evaluation logic without needing actual audio files.

⸺

## 🎧 Audio understanding (real interviews)

The system transcribes `.mp3` interview recordings using Gemini, with bilingual instructions and optional speaker tagging. Audio files are uploaded and passed as input:

    audio_path = "/kaggle/input/your‑dataset‑name/your_audio.mp3"
    uploaded_audio = client.files.upload(file=audio_path)

    transcribe_instruction = {
        "en": "Transcribe the following audio in English. If you can identify the speaker's name, use it. If not, use labels like 'Speaker 1:', etc."
    }[NOTEBOOK_UI_LANGUAGE]

    transcribe_response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=[transcribe_instruction, uploaded_audio],
        config=types.GenerateContentConfig(temperature=0.2)
    )

The result is a clean transcript with speaker turns, suitable for semantic chunking and search.

⸺

## 🧠 Embeddings & vector search

The system embeds transcript chunks using Gemini’s `text-embedding-004` model, with batching and retry logic. ChromaDB stores and retrieves results semantically:

    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True
        @retry.Retry(predicate=is_retriable)
        def __call__(self, input: Documents) -> Embeddings:
            task = "retrieval_document" if self.document_mode else "retrieval_query"
            ...
            response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(task_type=task),
            )

    documents = [chunk["text"] for chunk in chunk_data]

    embed_fn = GeminiEmbeddingFunction()
    db = chromadb.Client().get_or_create_collection(
        name="familymemories",
        embedding_function=embed_fn
    )

    db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

⸺

## 📚 Retrieval-augmented generation (RAG)

Relevant chunks are pulled from ChromaDB and added to the prompt for Gemini to generate responses:

    for doc in retrieved_docs:
        doc_text = doc.replace("\n", " ")
        ref_label = {
            "es": "TEXTO DE REFERENCIA",
            "en": "REFERENCE TEXT"
        }[NOTEBOOK_UI_LANGUAGE]
        prompt += f"\n{ref_label}: {doc_text}"

The prompt itself explicitly tells Gemini to **answer using only the context provided**, and to do so with care and empathy if the information isn’t sufficient:

    "Answer the following question using only the context provided below. [...] If the context does not contain enough information, say so with care and empathy."

⸺

## 🧭 Evaluation

To assess response quality, the system asks Gemini to evaluate each answer in two steps: first, with a structured explanation (groundedness, fluency, and comment), and second, with a single overall score using a defined enum schema.

    class AnswerRating(enum.Enum):
        VERY_GOOD = '5'
        GOOD = '4'
        OK = '3'
        BAD = '2'
        VERY_BAD = '1'

Gemini is prompted like this:

> Evaluate the quality of an AI-generated response based on a question and context.  
>  
> **Criteria:**  
> - Groundedness: Is the answer based on the context?  
> - Fluency: Is the answer natural, clear, and coherent?  
>  
> Use the following JSON format:  
>  
> ```json  
> {  
>   "groundedness": [1-5],  
>   "fluency": [1-5],  
>   "comment": "Brief comment in English"  
> }  
> ```  
>  
> Then return a **single number** from 1 to 5 as the final rating.

    structured_config = types.GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema=AnswerRating
    )
    enum_response = chat.send_message(
        message="Give me only the final score (1–5).",
        config=structured_config
    )

⸺

## 📎 Few-shot prompting (optional)

The system includes optional few-shot prompting for synthetic interview generation. These examples are commented out by default, but can help shape tone and structure.

Example:

> Granddaughter: Grandma, what was your house like when you were a kid?  
> Grandma: We lived in a small two-bedroom house on the outskirts of town. It had a big front porch, and we didn’t have air conditioning—just fans in the windows. I shared a room with my sister, and our closet was always packed with hand-me-downs.  
> Granddaughter: What did your daily routine look like back then?  
> Grandma: I’d wake up early to help pack lunches for school. We walked about twenty minutes, rain or shine. After school, I had chores—feeding the chickens, helping with dinner.  
> Granddaughter: Do you remember what kinds of meals you had most often?  
> Grandma: Oh yes—lots of casseroles, meatloaf on Mondays, spaghetti on Fridays. My mom made a chicken pot pie from scratch that I’ve never been able to replicate.

⸺

## 🧩 Limitations

- **Chunking** is still basic — fixed-length and naive  
- **Top-k retrieval** can miss key context  
- **Prompt control** is fragile across languages or model updates  
- **Evaluations** vary slightly between runs  
- **Speaker info** is lost during embedding — re-tagging fix is in progress  
- **Long context window** usage not tested — future comparison to RAG

⸺

## 🧵 Conclusion

I hope this project empowers you to start chats with your loved ones. Through my own experience doing this kind of interviews I've been amazed at how much I've learned and the deeply meaningful moments shared with relatives when I've asked some seemingly simple questions.

If not today, when?

---

# 🫶 Recuerdos Familiares: Un Sistema Bilingüe para Explorar y Revivir Entrevistas
[Vélo en Kaggle →](https://www.kaggle.com/code/arturomp/family-memories)

Tengo varias entrevistas familiares grabadas — conversaciones casuales con familiares mayores — y quiero seguir haciendo más. Pero no recuerdo exactamente en qué parte se dijeron ciertas cosas. Reproducir horas de audio para encontrar un solo momento no es viable.

Así que construí un prototipo: algo que me permite buscar en esas entrevistas y volver a interactuar con ellas, en ambos idiomas. Me ayuda a revivir momentos, rastrear temas y formular nuevas preguntas para entrevistas futuras.

⸺

## 🧾 Generación controlada (entrevistas sintéticas)

Para probar el sistema o enriquecerlo sin usar datos personales, se pueden generar entrevistas sintéticas usando Gemini con prompts cuidadosamente diseñados. La generación está estrictamente controlada en formato, turnos y tono:

    transcript_prompt = {
        "es": f"""
            Imagina una conversación entre {speaker_list_es}.
            Asegúrate de que las personas más jóvenes hagan preguntas que den seguimiento o profundicen, y que las personas mayores den respuestas largas, con historias personales.
            Usa solo formato de diálogo estricto:
            Persona: línea
            Sin narración, sin resúmenes, sin biografías de personajes.
            La conversación debe tocar temas como infancia, migración, recetas, vida familiar y eventos importantes.
            Cada persona debe hablar entre 20 y 40 veces, con detalles naturales y creíbles.
        """
    }[NOTEBOOK_UI_LANGUAGE]

    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=transcript_prompt,
        config=types.GenerateContentConfig(temperature=0.9)
    )

El resultado es una conversación ficticia pero realista que se comporta como una entrevista real — útil para probar búsquedas, recuperación y evaluación sin usar audio real.

⸺

## 🎧 Transcripción de audio (entrevistas reales)

El sistema también puede transcribir grabaciones `.mp3` usando Gemini, con instrucciones bilingües y etiquetas opcionales de hablante:

    audio_path = "/kaggle/input/your‑dataset‑name/your_audio.mp3"
    uploaded_audio = client.files.upload(file=audio_path)

    transcribe_instruction = {
        "es": "Transcribe el siguiente audio en español. Si puedes identificar el nombre del hablante, úsalo. Si no, usa etiquetas como 'Hablante 1:', etc."
    }[NOTEBOOK_UI_LANGUAGE]

    transcribe_response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17",
        contents=[transcribe_instruction, uploaded_audio],
        config=types.GenerateContentConfig(temperature=0.2)
    )

El resultado es una transcripción con turnos de habla, lista para ser fragmentada y buscada semánticamente.

⸺

## 🧠 Fragmentación, embeddings y vector search

Los textos se dividen en fragmentos y se representan con embeddings de Gemini usando `text-embedding-004`. Luego se almacenan en ChromaDB para recuperación semántica:

    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True
        @retry.Retry(predicate=is_retriable)
        def __call__(self, input: Documents) -> Embeddings:
            ...
            response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(task_type=task),
            )

    embed_fn = GeminiEmbeddingFunction()
    db = chromadb.Client().get_or_create_collection(
        name="recuerdosfamiliares",
        embedding_function=embed_fn
    )

    db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

⸺

## 📚 Recuperación aumentada por generación (RAG)

Se recuperan fragmentos relevantes de ChromaDB y se agregan al prompt para que Gemini pueda generar respuestas basadas únicamente en ese contenido:

    for doc in retrieved_docs:
        doc_text = doc.replace("\n", " ")
        ref_label = {
            "es": "TEXTO DE REFERENCIA",
            "en": "REFERENCE TEXT"
        }[NOTEBOOK_UI_LANGUAGE]
        prompt += f"\n{ref_label}: {doc_text}"

El prompt también le indica explícitamente a Gemini que **solo use ese contexto**, y que si no es suficiente, lo diga con empatía:

    "Responde la siguiente pregunta usando solo el contexto proporcionado abajo. [...] Si no hay suficiente información, dilo con cuidado y empatía."

⸺

## 🧭 Evaluación automática

El sistema le pide a Gemini evaluar las respuestas generadas en dos pasos: primero con una explicación estructurada, y luego con una calificación global del 1 al 5.

Esto se configura con una clase enum:

    class AnswerRating(enum.Enum):
        VERY_GOOD = '5'
        GOOD = '4'
        OK = '3'
        BAD = '2'
        VERY_BAD = '1'

El prompt incluye un formato bilingüe como este:

> Evalúa la calidad de una respuesta generada por IA basada en una pregunta y un contexto.  
>  
> **Criterios:**  
> - Fundamentación: ¿Está basada en el contexto?  
> - Fluidez: ¿Es clara, natural y coherente?  
>  
> Usa el siguiente formato JSON:  
>  
> ```json  
> {  
>   "groundedness": [1-5],  
>   "fluency": [1-5],  
>   "comment": "Comentario breve en inglés"  
> }  
> ```  
>  
> Luego, entrega un **solo número** del 1 al 5 como calificación final.

Y se usa configuración estructurada para forzar la respuesta:

    structured_config = types.GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema=AnswerRating
    )
    enum_response = chat.send_message(
        message="Dame solo la calificación final (1–5).",
        config=structured_config
    )

⸺

## 📎 Few-shot prompting (opcional)

También se puede incluir un ejemplo corto en el prompt para dar forma al tono y estilo. Se deja comentado por defecto.

Ejemplo:

> Nieta: Abuela, ¿cómo era tu casa cuando eras niña?  
> Abuela: Vivíamos en una casa pequeña, con dos habitaciones, en las afueras del pueblo. Tenía un porche grande al frente y no teníamos aire acondicionado — solo ventiladores en las ventanas. Compartía cuarto con mi hermana y el clóset estaba lleno de ropa heredada.  
> Nieta: ¿Cómo era tu rutina diaria?  
> Abuela: Me levantaba temprano para ayudar con los almuerzos. Caminábamos veinte minutos a la escuela, lloviera o no. Después tenía que dar de comer a las gallinas y ayudar con la cena.  
> Nieta: ¿Qué comidas comían más seguido?  
> Abuela: Muchos guisos, pastel de carne los lunes, espagueti los viernes. Mi mamá hacía un pastel de pollo casero que nunca pude replicar.

⸺

## 🧩 Limitaciones

- **Segmentación** de fragmentos es básica (por longitud). Se puede mejorar usando estructura conversacional.  
- **Recuperación top-k** a veces falla — se podría reordenar o usar recuperación híbrida.  
- **Control del prompt** es frágil, especialmente al cambiar el idioma o la versión del modelo.  
- **Evaluación automática** es útil, pero a veces inconsistente entre ejecuciones.  
- **Información de hablante se pierde** al hacer embeddings — ChromaDB guarda solo texto. Se trabaja en reinsertar etiquetas de hablante al generar los fragmentos.  
- **Comparación con ventanas largas** está pendiente — actualmente se usa RAG, pero en el futuro se podría alimentar el transcript entero.

⸺

## 🌱 ¿A dónde podría ir esto?

El sistema podría extenderse a:

- Búsqueda por voz  
- Navegación por línea de tiempo  
- Modos de resumen o narración de recuerdos  
- Aportes colaborativos de varias personas

Pero incluso ahora, ya me ha ayudado a escuchar de otra manera — y a recordar más.

⸺

## 🧵 Conclusión

Espero que este proyecto te anime a iniciar charlas con tus seres queridos. En mi propia experiencia con este tipo de entrevistas, me ha sorprendido lo mucho que he aprendido y los momentos que he compartido con mis familiares cuando les he hecho preguntas aparentemente sencillas.

Si no hoy, ¿cuándo?