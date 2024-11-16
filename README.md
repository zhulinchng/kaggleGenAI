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

**Answer 1:** Two exciting features:

1. Grounding with Google Search (launched about two weeks ago), which helps ground LLM answers using Google's search index.
2. OpenAI compatibility (launched on Friday), allowing developers using OpenAI SDKs and libraries to try Gemini models with minimal code changes.

**Question 2:** Tell me a little bit about those two Flash series of models, and how you can do so much so quickly with a very small cost footprint.

**Answer 2:** The Flash API model is the smallest Gemini hosted model (8 billion parameters). It offers a very low cost per token (2-3 cents per million tokens, or 1 cent with cached tokens), making it cost-effective for developers to build AI-powered applications without worrying about high costs.

**Question 3:** What has been your favorite application for these kind of multimodal output scenarios, and how do you think about coupling those with Gemini APIs?

**Answer 3:** While it's still early to determine the real production use cases of multimodal output, I'm excited about bringing text to life using audio and video.  Converting written content into audio or video formats can significantly enhance the user experience.  Imagine, for example, bringing a large corpus of text documents to life in video form. Models like Imagen and Vo will be crucial in enabling this.

**Question 4:** How does the Gemini app use RHF to help improve its responses?

**Answer 4:** LLMs are fine-tuned using two steps: Supervised Fine-Tuning (SFT) with high-quality human-generated data, and Reinforcement Learning with Human Feedback (RLHF). RLHF aligns models to human preferences by using a reward model to penalize bad responses and reward good ones. This reward model is trained on human preference data, including user feedback (thumbs up/thumbs down) from the Gemini app, leading to improved performance over time.

**Question 5:** Since large language models learn from massive sources of data, do large language models simply interpolate within their training data or can they go beyond to make new discoveries?

**Answer 5:** LLMs can go beyond their training data to make new discoveries, primarily through techniques like search.  One example is [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), where an LLM was used to search for solutions to complex problems in computer science and mathematics. The LLM proposed solutions, an evaluator provided feedback, and an evolutionary algorithm selected the best solutions, which were then fed back to the LLM for improvement. This iterative process led to the discovery of new solutions not present in the training data. This concept is also known as test-time compute or inference scaling.

**Question 6:** Can larger language models be used to train smaller ones?

**Answer 6:** Yes, using a technique called distillation.  There are three main types:

1. Data distillation: The larger model generates synthetic training data for the smaller model.
2. Knowledge distillation: Aligns the token distribution of the smaller LLM with the larger one.
3. On-policy distillation: Uses the larger model as a "teacher" to score the smaller model's output within a reinforcement learning framework.

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

**Question 1:** Congratulations on winning one of TIME's inventions of the year for NotebookLM. Can you articulate some of the ways that NotebookLM has been impactful and kind of driving change in the generative AI space?

**Answer 1:** NotebookLM was built from the ground up with a state-of-the-art language model at its core. It's designed for thinking, organizing research, and writing.  It focuses on "source grounding" (similar to RAG), where uploaded documents serve as the basis for all interactions with the model. Inline citations allow for fact-checking and deeper dives into source material. This is all powered by Gemini Pro 1.5, leveraging its long context window and citation capabilities.

**Question 2:** What has been your favorite use case for NotebookLM?

**Answer 2:** Personally, using a single notebook with 8,000 quotes from books and articles read over the years, plus the text of written books (totaling 25 million words). This acts as an extension of memory, providing relevant connections and excerpts when exploring new ideas.  Another popular feature is audio overviews, which generates podcast-style conversations based on uploaded sources.  A fun use case is uploading a resume to generate an enthusiastic audio overview of accomplishments.

**Question 3:** How are you thinking about function calling and retrieval in the systems that you're building?

**Answer 3:** For NotebookLM, semantic retrieval is crucial, especially with large source sets exceeding the model's context window. Smart tools retrieve the most relevant passages and present them to the model with a custom prompt.  For Data Science Agent in Colab, function calling, retrieval, and other techniques are used to enhance user velocity and experimentation.

**Question 4:** Have you seen any major changes to agent compositions and production since you started writing the white paper? What's changed, what's stayed the same, and then what are the challenges and opportunities you see?

**Answer 4:** Architectures have largely remained consistent (models, tools, orchestration layers, runtimes, memory, goals).  The biggest change is the improvement in models themselves, with better tool calling, logic, reasoning, Chain of Thought, and code execution capabilities, simplifying the orchestration layer.  Opportunities lie in solving complex real-world problems across various industries (supply chain, security) using multiple APIs, mutating actions, and simulations, moving beyond simple chat implementations.

**Question 5:** Has anyone implemented an agent focused on security? Where can I find overall guidance on implementing agents for cybersecurity?

**Answer 5:** The security industry offers many opportunities for agents (threat classification, tracking).  A semi-agentic setup for security challenges using LLMs will be discussed in upcoming sessions.  (Specific implementations weren't mentioned).

**Question 6:** Is there a way to evaluate the accuracy of the tools that Gemini selects? If there are deviations whenever Gemini selects a tool, are there any ways to remediate?

**Answer 6:** Vertex AI's Gen Evaluation service can be used to evaluate tool selection accuracy.  Generally, log and analyze tool selection decisions, develop diverse test cases, refine prompt engineering, improve tool definitions (potentially using the model itself), and consider targeted learning (fine-tuning or RAG-based queries).

**Question 7:** In building an agentic RAG system for better responses, there are a lot of intermediate steps between the user query and getting a response from the large language model. How can we deal with that latency?

**Answer 7:** Invest in traditional application development best practices: pre-processing, data quality improvement, chunking mechanisms, search performance.  Execute multiple queries in parallel and select the most relevant answers.  Prioritize simplicity in system design, even if it means not using the latest generative AI techniques.  Leverage context caching and explore long context windows for optimization.

**Question 8:** When would you recommend using a minimal implementation like the Gemini API versus a stateful graph-based approach like using LangGraph?

**Answer 8:** Use the Gemini API for quick prototypes and early exploration. For more complex agentic systems, adopt a system like LangGraph (or similar graph-based frameworks like [Breadboard](https://github.com/breadboard-ai/breadboard)) early on to improve control, introspection, and observability. Graph-based representations help developers understand and manage agent behavior.

## Day 4 - Domain-Specific Models

> [Day 4 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](https://www.youtube.com/watch?v=odvuLMJWUSU)
> Parsed with Anthropic's Claude 3.5 Sonnet as quota was exhausted for Gemini.

**Question 1:** Tell me a little bit about large language models for security. What are they, and why are they useful?

**Answer 1:** Security is not one monolithic thing but involves many different tasks from binary code analysis and malware analysis to triaging alerts and network administration.
The challenges include:

1) Limited public data availability due to security sensitivity,
2) Data being highly sensitive for organizations,
3) Security tasks often triggering safety mechanisms.

Specialized security LLMs involve continued pre-training on expert security documents, specific fine-tuning tasks, and human alignment processes. They're particularly useful because generalist models have higher hallucination rates for security tasks and often can't handle specialized security query languages. The focus is on having models perform tasks on data rather than memorizing information, since threats evolve daily making memorization impractical.

**Question 2:** How are you thinking about benchmarks like MedQA, and when will we reach 100% there?

**Answer 2:** MedQA is a multiple choice medical test similar to medical license exams. While models like Gemini are now achieving 90% accuracy, there are limitations to this benchmark:

1) It uses compressed patient information rather than full health records,
2) Multiple choice format is unrealistic compared to real medical decision-making.

While 100% accuracy may be achieved, it's no longer considered an important milestone. The focus is shifting toward more sophisticated benchmarks that better reflect reality, such as medical imaging tasks and complex case challenges with full patient scenarios requiring diagnosis and treatment recommendations.

**Question 3:** What are the trade-offs between general purpose and fine-tuned models?

**Answer 3:** There's a need to balance multiple dimensions including answer quality, serving cost, and latency. Domain-specific models can achieve good gains while being smaller and more specialized. There are multiple approaches:

1) Fine-tuning - adapting existing models with domain-specific samples, though this is semi-static,
2) Dynamic information through context at inference time using techniques like retrieval augmentation,
3) In-context learning with caching to improve efficiency.

The choice depends on the specific use case, requiring balance across different dimensions.

**Question 4:** Will a single superior model solve all health problems?

**Answer 4:** No single model will solve all health problems. Healthcare has evolved over centuries with many innovations like clinical studies, vaccines, and preventive care. While AI can make existing processes more efficient, the real opportunity lies in enabling new pathways of care and scientific discoveries that weren't possible before. While incremental improvements are important, the focus should be on exploring completely new possibilities that weren't conceivable before models like Gemini or Med-PaLM. Healthcare improvement is a lifetime journey, but AI will help get us closer to better outcomes.

**Question 5:** Security is an inherently adversarial field. Does its adversarial nature change your strategies for where and how you deploy LLMs for security purposes?

**Answer 5:** The adversarial nature requires careful consideration of new attack surfaces. Key considerations include:

1) Data cleaning and ensuring accuracy of training data from external sources,
2) Addressing prompt injection threats, similar to SQL injection but more challenging due to natural language being both input and programming language,
3) Mitigating risks by decomposing problems into smaller parts to confine potential malicious inputs,
4) Using techniques like heuristic scanning of inputs and confining impacts to specific components rather than entire systems.

There's active research in developing various defensive techniques as there's no fundamental fix yet for these challenges.

**Question 6:** What are the ethical considerations when creating specialized LLMs for sensitive fields, like healthcare? How can we ensure that patient privacy and data security are maintained during training and deployment?

**Answer 6:** For deployment, compliance with local regulations like HIPAA is essential, and Google Cloud infrastructure is designed to be compliant with these frameworks. For training, they use de-identified or synthetic data, following standards for data anonymization including name removal and date shifting. This serves two purposes:

1) Ensuring patient privacy,
2) Improving model performance by preventing the model from learning individual outliers rather than general patterns.

The right level of abstraction through de-identification and proper aggregation is crucial for both model performance and preventing personal data leaks.

**Question 7:** What do you think is the most effective use of LLMs in solving security problems? Why?

**Answer 7:** The most effective uses include:

1) Performing security tasks through procedural knowledge rather than relying on factual knowledge,
2) Acting as connective tissue between diverse security systems and data silos,
3) Code analysis and generation, particularly for fixing security misconfigurations,
4) Automating workflows while maintaining flexibility to adapt to different organizational contexts.

LLMs excel at leveraging APIs and tools to combine heterogeneous data in coherent ways, making them particularly effective at bridging gaps between different security systems and processes.

**Question 8:** What is one major gap that you see in current LLM capabilities that is vital for success in the security domain?

**Answer 8:** Two major gaps were identified:

1) Explainability and trust - security professionals are naturally skeptical and need better ways to verify model outputs without having to do as much work as they would have done originally. Current citation and grounding approaches still require significant human verification.
2) Tool use capabilities - current LLM tool use is rudimentary and requires extensive hand-tuning, especially when dealing with complex APIs that have hundreds or thousands of fields. The ability to effectively understand and use pre-existing tools in a scalable way would be game-changing for security applications.

**Question 9:** I saw the Med-PaLM evaluation whitepaper on diagnosing depression and PTSD, using client interviews as the basis for its assessments. How has it evolved since then?

**Answer 9:** The work has evolved beyond just diagnosis from transcripts to focus on more comprehensive patient care. While diagnostic accuracy has improved, the focus has shifted to:

1) Helping patients understand and follow through with recommended treatments,
2) Incorporating patients' life situations into recommendations,
3) Developing systems like Med-PaLM to be interactive chat partners rather than just diagnostic tools.

While these are early explorations, the emphasis is on demonstrating real-life impact through clinical studies and ensuring positive patient outcomes in safe ways.

## Day 5 - MLOps for Generative AI

> [Day 5 Livestream with Paige Bailey – 5-Day Gen AI Intensive Course | Kaggle](https://www.youtube.com/watch?v=uCFW0i9xrBc)

**Question 1:** How has MLOps changed with the introduction of large language models and generative AI?

**Answer 1:** MLOps has evolved significantly from manual processes in early 2010s to today. Key changes include:

1) Incorporation of new roles like prompt engineers and AI engineers,
2) Broader artifact management including model configurations, foundation models, prompt templates, and chaining pipelines,
3) Shift towards holistic application monitoring rather than just model monitoring,
4) Integration of user feedback and task-specific evaluations,
5) Need for more agile workflows due to rapidly emerging models and frameworks.

The field has moved from manual deployments to automated pipelines and serverless machine learning, with cloud adoption accelerating scalability and reliability while reducing costs.

**Question 2:** Tell me about the importance of evaluation for productionizing models. Is it only possible for text or is multimodal evaluation also an option?

**Answer 2:** Evaluation has become more complex with generative AI. For text evaluation, traditional metrics like BLEU or ROUGE scores can be used when comparing against ground truth, but they have limitations. LLMs can now be used as judges to evaluate other models' responses.

For multimodal evaluation, particularly in image and video generation, new approaches are being developed.

The [Gecko](https://arxiv.org/pdf/2403.20327) work at Google DeepMind demonstrates how Gemini models can be used to break down evaluation tasks into sub-questions for better understanding of model performance. Key considerations include:

1) Defining clear use cases and building comprehensive evaluation datasets,
2) Moving beyond single scores to provide more detailed breakdowns,
3) Giving users control and flexibility in evaluation metrics,
4) Developing general approaches that can work across different modalities.

**Question 3:** What MLOps challenges are no longer priorities, given many companies are now using REST API calls as opposed to training, deploying, and maintaining their own models?

**Answer 3:** Several traditional MLOps challenges have become less critical:

1) Data preparation, training, and loss function evaluation are now handled by model providers,
2) Model deployment and auto-scaling concerns are managed by the API providers,
3) Traditional data and model drift monitoring is less relevant.

However, new priorities have emerged:

1) Focus on evaluation of toxicity, factual knowledge, and task-specific metrics,
2) Need for different types of guardrails focused on use cases rather than numerical thresholds
3) Shift from data science skills to prompt engineering and AI engineering skills.

Companies can now focus more on building applications rather than managing infrastructure.

**Question 4:** Tell me a little bit about Vertex AI, and why a customer might want to use it at their company?

**Answer 4:** Vertex AI is an Enterprise AI platform that serves as a one-stop shop for both generative and non-generative machine learning tasks. Key features include:

1) Access to multiple models including Gemini, Anthropic, and other open-source models,
2) Tools for grounding models and building agents,
3) Complete MLOps toolkit including managed notebooks, experiment tracking, and MLOps pipelines
4) AutoML capabilities with point-and-click interface for users with less ML expertise.

The platform abstracts away infrastructure management, allowing companies to focus on ML implementation rather than DevOps concerns.

**Question 5:** What specific MLOps practices should be prioritized when starting with generative AI on Vertex AI, and how do these MLOps workflows for predictive practices differ from traditional models?

**Answer 5:** Key MLOps practices for generative AI include:

1) Model Discovery - systematically evaluating models based on quality, latency, cost, and compliance,
2) Prompt Engineering - managing prompts as both data and code with version control and testing,
3) Chaining and Augmentation - integrating external APIs and data sources,
4) Model Tuning and Training - supporting fine-tuning techniques and artifact management,
5) Evaluation - implementing both manual and automated approaches,
6) Deployment - managing components with standard software engineering practices,
7) Governance - establishing control and transparency over the entire lifecycle.

Beginner-friendly tools include Vertex AI Model Garden, Vertex Studio playground for experimentation, Agent Builder for creating conversational chatbots, and Vertex AI pipelines for MLOps automation.

**Question 6:** What are some strategies to monitor large generative AI models using Google Cloud in production environments with high variability in user queries?

**Answer 6:** Monitoring strategies focus on two main aspects:

1) System Performance - tracking latency, query size, and other operational metrics,
2) Model Quality - monitoring answer accuracy, relevance, and safety.

Implementation strategies include:

1) Storing detailed logs of user queries, retrieved snippets, prompts, and generated answers in BigQuery,
2) Creating embeddings for queries and answers to group similar ones together,
3) Analyzing user feedback for specific query clusters,
4) Using Looker integration for building monitoring dashboards,
5) Leveraging open-source observability tools coupled with Google Cloud Trace and Logging for detailed tracing of model interactions.

**Question 7:** How does Vertex AI enhance MLOps for foundation models and generative AI applications? Are there specific features that make it more suited for generative AI compared to other tools?

**Answer 7:** Vertex AI enhances MLOps for generative AI through several features:

1) Prompt Optimization Tool - automating prompt engineering and optimization using evaluation datasets,
2) Comprehensive Evaluation Tools - comparing models, prompts, and measuring fine-tuning impact,
3) Production Monitoring - experimental capabilities to evaluate model performance and safety compliance,
4) Integration with other GCP tools - including pipelines and experiments for scalable experimentation,
5) Prompt Management - storing, understanding, and iterating on prompts,
6) Prompt Gallery - providing starting templates for various applications.

These features are designed to scale effectively for enterprise needs while maintaining quality and safety standards.
