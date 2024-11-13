# Q&A Session for Kaggle's  5-Day Gen AI Intensive Course with Google

> Livestream transcripts parsed with Gemini 1.5 Pro, with manual editing where required. If you spot an error, do open a PR. System prompts are available [here](system_prompt.txt).

Course: [5-Day Gen AI Intensive Course with Google](https://rsvp.withgoogle.com/events/google-generative-ai-intensive)

- [Q\&A Session for Kaggle's  5-Day Gen AI Intensive Course with Google](#qa-session-for-kaggles--5-day-gen-ai-intensive-course-with-google)
  - [Day 1 - Foundational Large Language Models \& Prompt Engineering](#day-1---foundational-large-language-models--prompt-engineering)
  - [Day 2 - Embeddings and Vector Stores](#day-2---embeddings-and-vector-stores)
  - [Day 3 - AI Agents](#day-3---ai-agents)
  - [Day 4 - Domain-Specific Models](#day-4---domain-specific-models)
  - [Day 5 - MLOps for Generative AI](#day-5---mlops-for-generative-ai)

## Day 1 - Foundational Large Language Models & Prompt Engineering

> [Day 1 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](<https://www.youtube.com/watch?v=kpRyiJUUFxY>)

**Question 1:** Tell me which features you're most excited about, and which have already launched or about to launch?

**Answer 1:** Two exciting features: 1. Grounding with Google Search (launched about two weeks ago), which helps ground LLM answers using Google's search index. 2. OpenAI compatibility (launched on Friday), allowing developers using OpenAI SDKs and libraries to try Gemini models with minimal code changes.

**Question 2:** Tell me a little bit about those two Flash series of models, and how you can do so much so quickly with a very small cost footprint.

**Answer 2:** The Flash API model is the smallest Gemini hosted model (8 billion parameters). It offers a very low cost per token (2-3 cents per million tokens, or 1 cent with cached tokens), making it cost-effective for developers to build AI-powered applications without worrying about high costs.

**Question 3:** What has been your favorite application for these kind of multimodal output scenarios, and how do you think about coupling those with Gemini APIs?

**Answer 3:** While it's still early to determine the real production use cases of multimodal output, I'm excited about bringing text to life using audio and video.  Converting written content into audio or video formats can significantly enhance the user experience.  Imagine, for example, bringing a large corpus of text documents to life in video form. Models like Imagen and Vo will be crucial in enabling this.

**Question 4:** How does the Gemini app use RHF to help improve its responses?

**Answer 4:** LLMs are fine-tuned using two steps: Supervised Fine-Tuning (SFT) with high-quality human-generated data, and Reinforcement Learning with Human Feedback (RLHF). RLHF aligns models to human preferences by using a reward model to penalize bad responses and reward good ones. This reward model is trained on human preference data, including user feedback (thumbs up/thumbs down) from the Gemini app, leading to improved performance over time.

**Question 5:** Since large language models learn from massive sources of data, do large language models simply interpolate within their training data or can they go beyond to make new discoveries?

**Answer 5:** LLMs can go beyond their training data to make new discoveries, primarily through techniques like search.  One example is [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), where an LLM was used to search for solutions to complex problems in computer science and mathematics. The LLM proposed solutions, an evaluator provided feedback, and an evolutionary algorithm selected the best solutions, which were then fed back to the LLM for improvement. This iterative process led to the discovery of new solutions not present in the training data. This concept is also known as test-time compute or inference scaling.

**Question 6:** Can larger language models be used to train smaller ones?

**Answer 6:** Yes, using a technique called distillation.  There are three main types: 1. Data distillation: The larger model generates synthetic training data for the smaller model. 2. Knowledge distillation: Aligns the token distribution of the smaller LLM with the larger one. 3. On-policy distillation: Uses the larger model as a "teacher" to score the smaller model's output within a reinforcement learning framework.

**Question 7:** What are some approaches to evaluating large models?

**Answer 7:** Evaluating LLMs depends on the task and definition of "evaluation." Classical metrics like BLEU and ROUGE scores compare generated text to a ground truth. However, this isn't always feasible. LLMs can also be used as auto-raters, evaluating other LLMs on a pointwise or pairwise basis. Vertex AI and open-source libraries like [Promptfoo](https://github.com/promptfoo/promptfoo) offer tools for both human and auto-evaluation.

**Question 8:** Can someone please explain why for the first Chain of Thought prompt it's also explaining step by step instead of answering that directly?

**Answer 8:** Chain of Thought prompting encourages LLMs to generate intermediate reasoning steps. The issue in the example arose because the initial prompt inadvertently included intermediate steps, making it difficult to demonstrate the difference between standard and Chain of Thought prompting.  Changing parameters like top-k, top-p, or temperature, or using older Gemini models, might help reproduce the desired behavior.

**Question 9:** Is something like enum mode a fine-tune model designed to return enum values?

**Answer 9:** It depends on the goal. If fine-tuning, structure the input data with questions/tasks and output data with corresponding enum values. LLMs can also handle enums zero-shot.  Tell the LLM the available enum values and ask it to choose the appropriate one for a given question. Try zero-shot first, then fine-tune if necessary.

**Question 10:** How was NotebookLM built and created?

**Answer 10:** NotebookLM uses Gemini 1.5 Pro and 1.5 Flash, along with special techniques related to retrieval and careful prompting, rather than a dedicated fine-tuned version of Gemini.

## Day 2 - Embeddings and Vector Stores

> [Day 2 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](https://www.youtube.com/watch?v=86GZC56rQCc)

**Question 1:** Tell me a little bit about Vector databases and embeddings, what are they and why are they useful?

**Answer 1:** Embeddings are numerical vectors. We use embedding models, which are machine-learned models that convert real-world objects such as text, image, and video to embeddings. These models are trained so that the geometric distance or the similarity of the embeddings reflects the real-world similarity of the object that they represent. Embedding models can be used in recommendation systems, semantic search, classification, ranking, and many other applications. Google's Vertex AI platform offers two classes of pre-trained embedding models: text embedding models and multi-modal embedding models. Text embedding models allow you to convert plain text to embeddings, while multi-modal ones work with image, audio, and video input in addition to textual input. To use these models, you simply enable Vertex AI in your GCP project and send requests to Vertex endpoint. Vector databases are storage solutions that specialize in managing these embedding vectors.

**Question 2:** What are the trade-offs between using an open-source Vector database versus a proprietary one like some of the managed services that were just being mentioned on Google Cloud?

**Answer 2:** Open-source vector databases may be more favorable for aspects such as cost, flexibility, customizability, community support, and potentially avoiding vendor lock-in. However, this often needs to be balanced with things such as higher maintenance costs, higher management costs and complexity, and potential for fragmentation down the road, along with limited support options. Proprietary vector databases often have an edge in terms of ease of use as a more managed service, support, more advanced features and stability. But the downsides can be cost, potential for vendor lock-in in some cases, and limited customization flexibility and sometimes transparency. Increasingly powerful vector search and indexing capabilities are being added across various general-purpose databases and data analytics and warehouse platforms as first-class citizens. This spans both open source and proprietary worlds. For example, PG Vector extensions for Postgres. In the case of Google Cloud, AlloyDB, Spanner, and BigQuery are all adding very strong capabilities as first-class citizens. The advantage of this option is avoiding data duplication across two databases, having a single source of truth, and continuing to benefit from decades of work building highly advanced capabilities into these general-purpose databases and data warehouses.

**Question 3:** Do you think new features and capabilities like Gemini's longer context windows, which are now up to 2 million tokens in context, and Google's recently released search grounding will reduce the need for Vector databases, or are these kinds of features and tools complimentary to each other?

**Answer 3:** Longer context windows are exciting, but we are not ready to use them at a very large scale. First, current support for a few million tokens is still smaller than most realistic databases or corpora, which often need billions or trillions of tokens. Also, it's computationally expensive to run the full LLM over a massive amount of tokens, whereas vector databases are extremely efficient and can retrieve from billions of items very quickly. Finally, as more content is put into the context, the LLMs' reasoning capability starts to degrade. So it's often more useful to first use the vector database to retrieve the most relevant documents or content and then perform reasoning on things actually relevant to the user question. Vector databases are a complimentary technique to long context language model capabilities. The initial retrieval stage doesn't have to be extremely precise; instead, we can focus on recall, retrieving lots of relevant documents, putting them in context, and then the LLM can reason over that large amount of context. Search grounding allows grounding to the web and all public information, but often search or grounding is needed on private information like personal data or a company's proprietary private corpus, and for that, vector databases are still needed because public search will not have access to that private data.

**Question 4:** What are the fundamental challenges and opportunities for Vector databases?

**Answer 4:** From the perspective of building a vector search index natively within an operational database, one challenge is how to achieve the same performance of specialized vector database systems that don't have the same constraints as a system of record, where users expect transactional semantics, strong consistency, fast transactional operations, and data that spans memory and disk. One example of innovation is AlloyDB's custom underlying buffer page format to leverage scan's low-level optimizations. Other opportunities for vector databases are in improving usability and performance when vector search is only one part of the application logic. For example, vector search with filtering is a well-known challenge where it may be more efficient to do post-filtering or pre-filtering depending on the filtering condition. Vector search databases should be able to automatically figure out the actual filtering operation that will give the best performance without the user needing to specify that. This extends to combining with other application logic like joins, aggregation, and text search.

**Question 5:** What happens if retrieval augmented generation doesn't retrieve your relevant documents or your relevant data assets? What can you do?

**Answer 5:** It's not uncommon for retrieval systems to retrieve irrelevant documents, especially if the relevant information isn't present in the corpus.  This relies heavily on the backend large language model's factuality capabilities to determine if retrieved documents can answer the question. Other methodologies can ensure RAG systems function well even with irrelevant documents. One approach is tuning embeddings to improve the search aspect of RAG, which in turn improves generation. Another is using an agentic system instead of simple RAG. An AI agent dynamically figures out the best approach based on the query and prompt, using various tools and techniques like calling different vector databases or grounding services to retrieve the most relevant content for the prompt.

**Question 6:** The mainstream of LLMs are decoder models. How can we train an embedding model from a decoder-only model backbone?

**Answer 6:** Traditional LLMs are decoder-only, meaning they predict the next token by looking only at preceding tokens. This unidirectional, left-to-right causal attention allows for efficient training and optimization. However, for encoder-style tasks like creating embeddings, bidirectional attention (where every word considers surrounding words) is helpful. State-of-the-art embedding models initialize from decoder-only LLMs but are further trained for bidirectional attention, improving embedding quality.  They use encoder-style bidirectional attention but are initialized from decoder-only backbones with additional training.

**Question 7:** After all of this discussion, does this mean that conventional methods for creating embeddings will become obsolete?

**Answer 7:** Conventional methods, like OCR for extracting text from PDFs before using text embedding models, might eventually be replaced by future multimodal embedding models that directly recognize text from images. However, current multimodal embeddings aren't accurate enough to capture all text from images.  Combining both methods may be necessary for the next year or so to leverage the advantages of each.

## Day 3 - AI Agents

> [Day 3 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](https://www.youtube.com/watch?v=HQUtMWoTAD4)

## Day 4 - Domain-Specific Models

> [Day 4 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](https://www.youtube.com/watch?v=odvuLMJWUSU)

## Day 5 - MLOps for Generative AI

> [Day 5 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](https://www.youtube.com/watch?v=uCFW0i9xrBc)
