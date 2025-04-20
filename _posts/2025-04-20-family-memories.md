---
layout: post
title: "Family Memories: Searchable Interviews with RAG"
subtitle: "A bilingual pipeline for recalling stories, building questions, and exploring shared history"
categories: nlp llm
permalink: /projects/family-memories-rag.html
---

# 🫶 Family Memories: A Bilingual AI System to Explore and Revisit Family Interviews  
[View on Kaggle →](https://www.kaggle.com/code/arturomp/family-memories)

(🇲🇽 en español más abajo 👇)

I have several recorded family interviews: conversations where I ask older relatives about their lives. They are incredibly valuable to me, to them and to our entire family, and I plan to do many more for as long as I can. But since this is a years-long effort, it's hard to remember what things were said and by whom. Replaying hours of audio to find one moment isn’t sustainable.

So I built a system to help me: it makes those interviews searchable and easier to interact with, across languages. It helps me zoom in and revisit moments or zoom out and look at a greater picture in their narratives.

I implemented this in a [Kaggle notebook](https://www.kaggle.com/code/arturomp/family-memories). In this post I focus on some implementation bits that are particularly interesting. 

However, I do try to make the whole notebook as easy-to-follow as possible, so check the whole thing out [in Kaggle](https://www.kaggle.com/code/arturomp/family-memories)!

⸺

## 🧾 Controlled generation (synthetic interviews)

To simulate interviews (for testing and demoing without using personal data), the system uses Gemini to generate synthetic transcripts using carefully constructed prompts. The generation is tightly controlled via format, speaker turns, and tone:

    transcript_prompt = {
        ...
        "en": f"""
            Imagine a conversation between {speaker_list_en}.
            Make sure that younger people ask questions that advance the interview by either (i) clarifying or seeking to elaborate on what they heard in response from older people, or (ii) asking a new question without switching topics abruptly, that is, without ignoring or not acknowledging what was said.
            Make sure that older people have long turns, each varying from 5 sentences sometimes up to 15 sentences about their life and experiences.
            Only write the dialogue — no narration, headings, or character descriptions, inner dialogue or parentheses.  
            Do not include scene-setting, stage directions, speaker bios, or turn counts.
            As in a real interview, interviewees should tell stories with characters to provide context. The stories told should not contradict each other within the interview, although another participant may have a different perspective on the same story.
            Strict format: 
            Each line should begin with the speaker’s name, followed by a colon and their line.  
            Example:  
            Grandma: [grandma-turn]
            Dad: [dad-turn]
                    Topic:  
            This is an interview-style conversation: a younger person (like a daughter or granddaughter) asks older family members questions about their life.  
            Topics include childhood, family recipes, everyday life in earlier times, traditions, migration, education, work, religion, births, deaths, and how family members met one another.
            Additional instructions:
            - The conversation should feel natural and include personal (but fictional) details.
            - Each person should speak from 20 to 40 times, in no particular order.
        """
    ...
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17", # try 2.5! Better synthetic stories.
        # model="models/gemini-2.0-flash", 
        contents=transcript_prompt,
        config=types.GenerateContentConfig(temperature=0.9)
    )

The result is a fictional yet realistic family conversation that behaves just like real input — useful for testing search, retrieval, and evaluation logic without needing actual audio files.

⸺

## 🎧 Audio understanding (real interviews)

The system transcribes `.mp3` interview recordings using Gemini, with bilingual instructions and optional speaker tagging. Audio files are uploaded and passed as input:

    audio_path = "/kaggle/input/your‑dataset‑name/your_audio.mp3"
    uploaded_audio = client.files.upload(file=audio_path)
    
    ...
    
        transcribe_instruction = {
        ...
        "en": (
            "Transcribe the following audio in English. "
            "If you can identify the speaker's name, use it. If not, use labels like 'Speaker 1:', 'Speaker 2:', etc. "
            "Format each turn as 'Speaker: sentence'. "
            "Only return the transcript — no explanations or headings."
        )

Model choice can make a big difference:

    transcribe_response = client.models.generate_content(        
        model="models/gemini-2.5-flash-preview-04-17", # try 2.5! Much more advanced (speaker identification/diarization!) and even translated transcripts.
        # model="models/gemini-2.0-flash",
        contents=[
            transcribe_instruction,
            uploaded_audio
        ],
        config=types.GenerateContentConfig(temperature=0.2)
    )

The result is a clean transcript with speaker turns, suitable for semantic chunking and search.

One of my favorite aspects of this feature is that since modern speech reco systems like Gemini can work with dozens of languages, your interviews can be in [~40](https://ai.google.dev/gemini-api/docs/models#supported-languages) languages supported by Gemini.

Today, different generations of the same family often have different primary languages due to migration, education and many other factors. Powerful, multilingual transcription allows each member to tell their own stories in the language in which they are most comfortable. Our system then allows any other family member, who may or may not share their primary language, to understand more fully those stories. This should have the effect of fostering closer, more authentic connections.

⸺

## 🧠 Embeddings & vector search

The system embeds transcript chunks using Gemini’s `text-embedding-004` model, with batching and retry logic.

    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    
    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True  # Switch to False for queries
    
        @retry.Retry(predicate=is_retriable)
        def __call__(self, input: Documents) -> Embeddings:
            task = "retrieval_document" if self.document_mode else "retrieval_query"
    
            def batch_iterable(iterable, batch_size):
                for i in range(0, len(iterable), batch_size):
                    yield iterable[i:i + batch_size]
    
            all_embeddings = []
    
            for batch in batch_iterable(input, 100):  # Gemini API limit: 100 per batch
                try:
                    response = client.models.embed_content(
                        model="models/text-embedding-004",
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task),
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])

ChromaDB stores and retrieves results semantically.

    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True  # embedding chunks for retrieval
    
    db = chroma_client.get_or_create_collection(
        name=DB_NAME,
        embedding_function=embed_fn
    )
    
    try:
        db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

⸺

## 📚 Retrieval-Augmented Generation (RAG)

Relevant chunks are pulled from ChromaDB and added to the prompt for Gemini to generate responses:


    result = db.query(query_texts=[query], n_results=RETRIEVAL_TOP_K)
    [retrieved_docs] = result["documents"]
    
    ...
    
    for doc in retrieved_docs:
        doc_text = doc.replace("\n", " ")
        ref_label = {
            "es": "TEXTO DE REFERENCIA",
            "en": "REFERENCE TEXT"
        }[NOTEBOOK_UI_LANGUAGE]
        prompt += f"\n{ref_label}: {doc_text}"

The prompt itself explicitly tells Gemini to **answer using only the context provided**, and to do so with care and empathy if the information isn’t sufficient:

    prompt = {
    ...
    "en": f"""Answer the following question using only the context provided below.  
            Imagine you are helping someone understand and recall family memories shared by someone else —such as a grandparent or relative who was interviewed.  
            If the context does not contain enough information, say so with care and empathy.  
            Be clear, respectful, and kind in your answer.
            
            QUESTION: {query_oneline}
            """

⸺

## 🧑‍⚖️ Automatic Evaluation

To assess response quality, the system asks Gemini to evaluate each answer in two steps: first, with a structured explanation (groundedness, fluency, and with a comment), and second, with a single overall score using a defined enum schema.

    class AnswerRating(enum.Enum):
        VERY_GOOD = '5'
        GOOD = '4'
        OK = '3'
        BAD = '2'
        VERY_BAD = '1'

Gemini is prompted like this:

    EVAL_PROMPT_TEMPLATE = {
    ... 
        "en": """\
        Evaluate the quality of an AI-generated response based on a question and context.
        
        ## Criteria:
        - Groundedness: Is the answer based on the context?
        - Fluency: Is the answer natural, clear, and coherent?
        
        Rate each from 1 to 5 and include a brief comment. Use the following JSON format:
        
        {{
          "groundedness": [1-5],
          "fluency": [1-5],
          "comment": "Brief comment in English"
        }}
        
        Then, following this rubric, return a **single number** from 1 to 5 as the overall rating:
        
        - 5 (VERY GOOD): if both scores are 5  
        - 4 (GOOD): if both are at least 4  
        - 3 (OK): if both are at least 3  
        - 2 (BAD): if either score is 2  
        - 1 (VERY BAD): if either score is 1
        
        ---
        
        🎯 Original question: {question}
        
        📚 Retrieved context:
        {context}
        
        📝 Generated response:
        {response}
    """

And structured configuration is used to force the response:

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

The system includes a short example in the prompt to shape the overall tone and style of the interview. It is left commented out by default as the topics of the example and the short sentences influenced the generation too much and affected flow negatively. That said, they may be useful in certain contexts.

    # You can add the following example to the variable above to perform few-shot prompting. 
    # In my tests it prevents the interview from flowing well.
    # """
    #     Example:
    #     Granddaughter: Grandma, what was your house like when you were a kid?
    #     Grandma: We lived in a small two-bedroom house on the outskirts of town. It had a big front porch, and we didn’t have air conditioning—just fans in the windows. I shared a room with my sister, and our closet was always packed with hand-me-downs. We had a rotary phone in the kitchen, and everyone had to take turns using it. The backyard had a big oak tree we used to climb.
    #     Granddaughter: What did your daily routine look like back then?
    #     Grandma: I’d wake up early to help pack lunches for school. We walked about twenty minutes, rain or shine. After school, I had chores—feeding the chickens, helping with dinner. If there was time, we’d ride our bikes around the neighborhood or play hopscotch on the driveway. Saturdays were for laundry and church on Sundays.
    #     Granddaughter: Do you remember what kinds of meals you had most often?
    #     Grandma: Oh yes—lots of casseroles, meatloaf on Mondays, spaghetti on Fridays. My mom made a chicken pot pie from scratch that I’ve never been able to replicate. And we always had Jell-O in the fridge. Every Sunday after church, we’d have a roast with mashed potatoes and green beans. Those were some of my favorite meals growing up.
    # """

⸺

## 🧩 Limitations

A kaggle notebook is a starting point, so limitations are plenty. However, here are some that are of immediate concern if you want to use this on your own data:


- **Embedding** unexpectedly lacks speaker awareness despite keeping that information in the chunks (ChromaDB seems to be the culprit, but unclear exactly how yet) - this can affect retrieval when speaker attribution is important
- **Evaluations** vary slightly between runs, but a stronger prompt may help
- **RETRIEVAL_TOP_K** during querying can miss key context if it's too short 

⸺

## 🔭 Future Directions


The system can be expanded with new functions and adaptations. Some current ideas are:

- **Interview assistant:** suggest context-aware follow-up questions during live interviews.  
- **Reusable flow:** adapt the pipeline to other interview corpora, such as [NPR Dialog Transcripts on Kaggle](https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts/data).  
- **Personal Oral History Summarizer:** segment interviews into chapters like *Childhood*, *Migration*, *Raising a Family*.  
- **Thematic Indexer:** use embeddings or agents to tag recurring themes (e.g., *resilience*, *food*, *gender roles*) across interviews.  
- **Quote Collage Generator:** surface moving, funny, or surprising quotes and format them for a family memory book.


⸺

## 🧵 Conclusion

I hope this project empowers you to start chats with your loved ones about their lives. Through my own experience doing this kind of interviews I've been amazed at how much I've learned and the deeply meaningful moments shared with relatives when I've asked some seemingly simple questions.

If not today, when?

---

# 🫶 Recuerdos Familiares: Un Sistema Bilingüe para Explorar y Revivir Entrevistas
[Velo en Kaggle →](https://www.kaggle.com/code/arturomp/family-memories)

Tengo varias entrevistas familiares grabadas: conversaciones en las que pregunto a parientes mayores sobre sus vidas. Son increíblemente valiosas para mí, para ellos y para toda nuestra familia, y pienso hacer muchas más mientras pueda. Pero como se trata de un proyecto que lleva años, es difícil recordar qué cosas se dijeron y quién las dijo. Reproducir horas de audio para encontrar un momento no es sostenible.

Por eso creé este sistema que permite buscar en esas entrevistas e interactuar con ellas más fácilmente en varios idiomas. Me ayuda a enfocarme en ciertos momentos o alejarme y ver una imagen más amplia de sus relatos.

Implementé el sistema en un [cuaderno de Kaggle](https://www.kaggle.com/code/arturomp/family-memories). Aquí me enfoco en algunas partes de la implementación del sistema que son especialmente interesantes.

Lo que sí es que intento que el cuaderno sea lo más fácil de seguir y usar posible, ¡así que échale un ojo [en Kaggle](https://www.kaggle.com/code/arturomp/family-memories)!

⸺

## 🧾 Generación controlada (entrevistas sintéticas)

Para simular las entrevistas (con fines de hacerle pruebas y hacer demos sin utilizar datos personales), el sistema utiliza Gemini para generar transcripciones sintéticas con indicaciones cuidadosamente elaboradas. 

La generación se controla estrictamente en la indicación (o _prompt_) mediante el formato, los turnos del orador y el tono:

    transcript_prompt = {
        "es": f"""
                Imagina una conversación entre {speaker_list_es}.
                Asegúrate de que las personas mas jóvenes hagan preguntas que avanzan la entrevista ya sea (i) aclarando o buscando llegar más a fondo sobre lo que escucharon como respuesta de los mayores, , o (ii) formulando una nueva pregunta sin cambiar bruscamente de tema, es decir, sin ignorar o no reconocer lo que se ha dicho.
                Asegúrate de que las las personas mayores tengan turnos largos, cado uno variando desde 5 enunciados a veces hasta 15 enunciados sobre su vida y sus experiencias.
                Escribe únicamente el diálogo, sin narración, encabezados ni descripciones, diálogo interior o paréntesis.
                No incluyas la escenografía, las instrucciones de escena, las biografías de los oradores ni el recuento de turnos.
                Como en una entrevista real, las personas entrevistadas deberían contar historias con personajes para dar contexto. Las historias contadas no deberían contradecirse dentro de la entrevista, aunque otro participante puede tener una perspectiva diferente de la misma historia.
                Formato estricto:  
                Cada línea debe comenzar con el nombre del hablante, seguido de dos puntos y su intervención.  
                Ejemplo:  
                Abuela: [turno-de-la-abuela]  
                Papá: [turno-del-papá]
                        Tema:  
                Es una conversación estilo entrevista: una persona más joven (como una hija o nieta) le hace preguntas a familiares mayores sobre su vida.  
                Tocan temas como la infancia, recetas familiares, el día a día en otras épocas, tradiciones, migraciones, educación, trabajo, religión, nacimientos, fallecimientos, y cómo se conocieron los miembros de la familia.
                Instrucciones adicionales:
                - La conversación debe sonar natural y estar llena de detalles personales (pero inventados).
                - Cada persona debe participar de 20 a 40 veces, sin un orden fijo.
                No agregues encabezados, personajes, narración, resúmenes ni cierres. Solo el diálogo limpio.
            """,
    ...
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-04-17", # try 2.5! Better synthetic stories.
        # model="models/gemini-2.0-flash", 
        contents=transcript_prompt,
        config=types.GenerateContentConfig(temperature=0.9)
    )

El resultado es una conversación ficticia pero realista que se comporta como una entrevista real — útil para probar búsquedas, recuperación y evaluación sin usar audio real.

⸺

## 🎧 Transcripción de audio (entrevistas reales)

El sistema también puede transcribir grabaciones `.mp3` usando Gemini, con instrucciones bilingües y etiquetas opcionales de hablante:

    audio_path = "/kaggle/input/your‑dataset‑name/your_audio.mp3"
    uploaded_audio = client.files.upload(file=audio_path)
    
    ...
    
        transcribe_instruction = {
        "es": (
            "Transcribe el siguiente audio en español. "
            "Si puedes identificar el nombre del hablante, úsalo. Si no, usa etiquetas como 'Hablante 1:', 'Hablante 2:', etc. "
            "Usa el formato 'Hablante: oración' para cada intervención. "
            "Responde únicamente con la transcripción, sin explicaciones ni encabezados."
        ),

La elección de modelo puede hacer una gran diferencia:

    transcribe_response = client.models.generate_content(        
        model="models/gemini-2.5-flash-preview-04-17", # try 2.5! Much more advanced (speaker identification/diarization!) and even translated transcripts.
        # model="models/gemini-2.0-flash",
        contents=[
            transcribe_instruction,
            uploaded_audio
        ],
        config=types.GenerateContentConfig(temperature=0.2)
    )

El resultado es una transcripción con turnos de habla, lista para ser fragmentada y para aplicarle búsqueda semántica.

Uno de mis aspectos favoritos de esta función es que, dado que los sistemas modernos de reconocimiento de voz como Gemini pueden trabajar con docenas de idiomas, sus entrevistas pueden ser en [~40](https://ai.google.dev/gemini-api/docs/models#supported-languages) idiomas admitidos por Gemini.

Hoy en día, distintas generaciones de una misma familia pueden tener diferente lenguas primarias gracias a factores como migración, educación y otros. El poder de la transcripción multilingüe permite a cada miembro contar sus propias historias en la lengua en la que se sientan más cómodos. Nuestro sistema permite entonces que cualquier otro miembro de la familia, sin importar que comparta o no su lengua primaria, comprenda mejor esas historias. Esto fomenta conexiones más auténticas y cercanas.

⸺

## 🧠 _Embeddings_ y búsqueda vectorial

El sistema hace embeddings con fragmentos de la transcripción utilizando el modelo `text-embedding-004` de Gemini, con lógica de procesamiento por lotes y reintentos.

    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    
    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True  # Switch to False for queries
    
        @retry.Retry(predicate=is_retriable)
        def __call__(self, input: Documents) -> Embeddings:
            task = "retrieval_document" if self.document_mode else "retrieval_query"
    
            def batch_iterable(iterable, batch_size):
                for i in range(0, len(iterable), batch_size):
                    yield iterable[i:i + batch_size]
    
            all_embeddings = []
    
            for batch in batch_iterable(input, 100):  # Gemini API limit: 100 per batch
                try:
                    response = client.models.embed_content(
                        model="models/text-embedding-004",
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task),
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])

Luego se almacenan los resultados en ChromaDB para recuperación semántica.

    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True  # embedding chunks for retrieval
    
    db = chroma_client.get_or_create_collection(
        name=DB_NAME,
        embedding_function=embed_fn
    )
    
    try:
        db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

⸺

## 📚 Generación Aumentada por Recuperación (RAG por sus siglas en inglés: _Retrieval-Augmented Generation_)

Se recuperan fragmentos relevantes de ChromaDB y se agregan a la indicación para que Gemini pueda generar respuestas basadas únicamente en ese contenido:

    result = db.query(query_texts=[query], n_results=RETRIEVAL_TOP_K)
    [retrieved_docs] = result["documents"]
    
    ...
    
    for doc in retrieved_docs:
        doc_text = doc.replace("\n", " ")
        ref_label = {
            "es": "TEXTO DE REFERENCIA",
            "en": "REFERENCE TEXT"
        }[NOTEBOOK_UI_LANGUAGE]
        prompt += f"\n{ref_label}: {doc_text}"

El prompt también le indica explícitamente a Gemini que **solo use ese contexto**, y que si no es suficiente, lo diga con empatía:

    prompt = {
    "es": f"""Responde la siguiente pregunta usando solo el contexto proporcionado abajo.  
            Imagina que estás ayudando a una persona a comprender y recuperar recuerdos familiares de otra persona —como un abuelo o familiar entrevistado—.  
            Si el contexto no contiene suficiente información, indícalo con cuidado y empatía.  
            Sé claro, respetuoso y amable en tu respuesta.
            
            PREGUNTA: {query_oneline}
            """,

⸺

## 🧑‍⚖️ Evaluación Automática

El sistema le pide a Gemini evaluar las respuestas generadas en dos pasos: primero con una explicación estructurada (fundamento, fluidez y con un comentario), y luego con una calificación global del 1 al 5, la que se configura con una clase enum:

    class AnswerRating(enum.Enum):
        VERY_GOOD = '5'
        GOOD = '4'
        OK = '3'
        BAD = '2'
        VERY_BAD = '1'

La indicación para la evaluación es la siguiente:


    EVAL_PROMPT_TEMPLATE = {
    "es": """\
        Evalúa la calidad de una respuesta generada por IA en base a una pregunta y contexto.
        
        ## Criterios:
        - Fundamentación: ¿Está la respuesta basada en el contexto?
        - Fluidez: ¿La respuesta suena natural, clara y coherente?
        
        Califica del 1 al 5 cada uno y proporciona un comentario breve. Usa el siguiente formato JSON:
        
        {{
          "groundedness": [1-5],
          "fluency": [1-5],
          "comment": "Comentario breve en español"
        }}
        
        Después, con base en estas reglas, responde solo con un número del 1 al 5 para representar la evaluación global:
        
        - 5 (MUY BUENO): si ambas puntuaciones son 5  
        - 4 (BUENO): si ambas son al menos 4  
        - 3 (OK): si ambas son al menos 3  
        - 2 (MALO): si alguna es 2  
        - 1 (MUY MALO): si alguna es 1
        
        ---
        
        🎯 Pregunta original: {question}
        
        📚 Contexto:
        {context}
        
        📝 Respuesta generada:
        {response}
        """,


Y se usa configuración estructurada para forzar la respuesta:

    structured_config = types.GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema=AnswerRating
    )
    enum_response = chat.send_message(
        message="Give me only the final score (1–5).",
        config=structured_config
    )

⸺

## 📎 Indicaciones con algunos ejemplos (_few-shot prompting_, opcional)

El sistema incluye un ejemplo corto en la indicación para dar forma al tono y estilo de la entrevista en general. Se deja comentado por defecto ya que los temas del ejemplo y las breves frases usadas influían demasiado la generación, además de afectar negativamente el flujo de la entrevista. Dicho eso, pueden ser útiles en ciertos contextos.

        # Puedes añadir el siguiente ejemplo a la variable arriba para que la indicación tenga 
        # algunos ejemplos (few-shot prompting). En mis pruebas hace que la entrevista no fluya tan bien.
        # """
        #         Ejemplo:
        #         Nieta: Abuela, ¿cómo era tu casa cuando eras niña?
        #         Abuela: Era una casa muy humilde, con techo de tejas y paredes de adobe. Teníamos solo dos habitaciones para siete personas. No había electricidad, y el agua la traíamos del pozo. Recuerdo que en invierno todo se sentía más cerca, porque nos reuníamos alrededor de la estufa de leña.
        #         Nieta: ¿Y cómo era tu rutina diaria en esa época?
        #         Abuela: Me levantaba antes del amanecer para ayudar a mi mamá a hacer las tortillas. Luego caminábamos casi una hora para llegar a la escuela. En las tardes ayudaba a cuidar a mis hermanos menores y a alimentar a los animales. Apenas si había tiempo para jugar.
        #         Nieta: ¿Cuál era tu juego favorito?
        #         Abuela: Jugábamos a la cuerda y a las escondidas. Pero mi favorito era inventar historias con mis amigas usando muñecas que hacíamos nosotras mismas con retazos de tela.
        # """

⸺

## 🧩 Limitaciones


Un cuaderno kaggle es un punto de partida, por lo que las limitaciones son muchas. Sin embargo, aquí hay algunas que son de interés inmediato si quieres utilizar esto en tus propios datos:



- Los **embeddings** inesperadamente pierden conocimiento sobre quién dice una oracióna pesar de mantener esa información en los fragmentos (ChromaDB parece ser el culpable, pero aún no está claro cómo exactamente), lo que puede afectar a la recuperación, sobre todo cuando la atribución del hablante es importante
- Las **evaluaciones** varían ligeramente entre ejecuciones, pero un aviso más fuerte puede ayudar.
- **RETRIEVAL_TOP_K** en la consulta puede perder contexto clave si tiene un valor bajos

⸺

##  🔭 Direcciones futuras

El sistema puede expandirse con nuevas funciones y adaptaciones. Algunas ideas actuales son:

- **Asistente de entrevistas:** sugerencias automáticas de preguntas de seguimiento durante entrevistas reales.  
- **Reutilización del enfoque:** aplicar este flujo a otros conjuntos de datos de entrevistas (como el de [NPR en Kaggle](https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts/data)). 
- **Resumen de historias orales:** segmentar entrevistas y resumirlas en capítulos como *Infancia*, *Migración*, *Criar una familia*.  
- **Indexador temático:** extraer y etiquetar temas frecuentes (como *resiliencia*, *comida*, *género*) en múltiples entrevistas.  
- **Generador de citas memorables:** crear collages con frases conmovedoras, graciosas o sorprendentes para incluir en un libro familiar.

⸺

## 🧵 Conclusión

Espero que este proyecto te anime a iniciar pláticas con tus seres queridos sobre sus vidas. En mi propia experiencia con este tipo de entrevistas, me ha sorprendido lo mucho que he aprendido y los momentos que he compartido con mis familiares cuando les he hecho preguntas aparentemente sencillas.

Si no hoy, ¿cuándo?