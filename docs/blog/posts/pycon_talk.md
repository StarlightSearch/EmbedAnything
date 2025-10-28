---
draft: false 
date: 2025-04-01
authors: 
 - sonam
slug: embed-anything
title: PyCon Germany
---
The 2025 PyCon DE event highlighted a growing but cautious interest in AI agents among the Python community. While agent technology received significant attention, many speakers and attendees expressed skepticism about their practical utility in real-world applications.

<!-- more -->

A notable trend was the gap between theoretical potential and actual implementation challenges. Blue Yonder teams, who work in supply chain management, shared how they overcome challenges with deploying agent-based solutions in production environments. Leonardo from Hugging Face presented a compelling mathematical limitation: even with models achieving 90% accuracy, a 20-step agent workflow would result in only about 12% end-to-end accuracy (due to compounding errors at each step). This mathematical reality presents a significant obstacle for complex agent workflows.

Alexander Hendorf raised essential questions about the actual use cases where agents provide meaningful value in practical applications, suggesting that the technology might still be searching for its most effective applications.

The conference reflected a Python community that remains interested in agent technology but is increasingly focused on practical implementation challenges rather than theoretical possibilities.

![image](https://royal-hygienic-522.notion.site/image/attachment%3A6a43f65c-8807-4e65-9125-962d1c95cc2e%3Aimage.png?table=block&id=1e281b6a-6bbe-80fc-9a9a-eac2c50c3580&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1420&userId=&cache=v2)

## Community & Culture

- Valerio Maggio moderated lightning talks described as "playful chaos" covering diverse topics.
- A Fireside Chat emphasized that technology should solve real problems, not exist for its own sake, and contrasted American startup culture with Germany's focus on sustainable development.
- Feminist AI Lan party, I had a boxing workshop, along with Ines and others. It was fun and frankly much needed after all the serious talks.

![conference](https://royal-hygienic-522.notion.site/image/attachment%3A792c99e7-fa4d-4e62-b852-86061b208da9%3Aimage.png?table=block&id=1e281b6a-6bbe-806e-b5c0-cab0b640a0d3&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=2000&userId=&cache=v2)
The conference reflected a Python community balancing excitement about new technology

PyConDE 2025: Building the Future of AI Agents - A Report from the Trenches

## **Beyond Agents: What AI Strategy Really Needs in 2025**

I was at a talk by [**Alexander C.S. Hendorf 👋**](https://www.linkedin.com/in/hendorf/) where he shared his insights on Nvidia GTC and shed light on a bunch of amazing topics , robotics , simulation, and open source. I appreciate some of the points he made regarding the commercialization of open-source software. And a deep question: "Are agents actually helpful?"

## Beyond Brute Force: Smarter Data Ingestion for RAG Systems

The first talk that really grabbed my attention was "PDFs - When a thousand words are worth more than a picture (or table)." The presenter highlighted a critical bottleneck in RAG (Retrieval Augmented Generation) systems: the inherent limitations of PDF parsing. We often take for granted the visual fidelity of PDFs, but their structure presents a real challenge for computers trying to extract meaningful information. Tables and figures, in particular, become nightmares for traditional parsers, leading to unreliable knowledge being fed into vector databases and ultimately, unreliable outputs from our RAG systems.

The solution? Embracing multimodal models. The speaker advocated for decomposing tables and figures into plain language descriptions, mimicking how a human would explain them. This approach, focusing on semantics rather than visual representation, promises to significantly improve retrieval accuracy and pave the way for more robust RAG-based agents. The key takeaway here is that building effective agents requires intelligent data ingestion strategies that go beyond simple text extraction.

## Federated Learning: Training Agents in a Privacy-Conscious World

Another compelling session, "The future of AI training is federated," addressed the growing need for privacy-preserving AI development. With increasing data privacy regulations (like GDPR) and logistical challenges in centralizing data, federated learning is emerging as a crucial paradigm. This talk provided a practical introduction to building FL systems using the open-source Flower framework.

The core concept is simple: instead of bringing the data to the model, we bring the model to the data. This allows us to train AI models on decentralized datasets without compromising sensitive information. The presenter walked us through converting a centralized ML workflow into a federated one, highlighting the specific steps involved in configuring clients, persisting state, and evaluating models. The biggest insight? Federated learning is no longer a niche research area; it's a practical solution for building AI agents that respect data privacy and comply with evolving regulations.

## MCP: Bridging the Gap Between Agents and Complex Systems

"From Idea to Integration: An Intro to the Model Context Protocol (MCP)" introduced a powerful standard for connecting Large Language Models with diverse data sources. The Model Context Protocol (MCP) acts as a bridge, enabling agents to interact with complex systems and access real-time data. The talk showcased how to build an MCP server and demonstrated its potential for empowering both developers and non-technical users. Imagine an agent that can access and interpret data from your smart home, your CRM, or any other complex system – that's the power of MCP.

The key takeaway here is that building intelligent agents requires seamless integration with the real world. MCP provides a standardized way to achieve this, opening up a world of possibilities for contextual AI applications.

## Beyond the Hype: Strategic AI in 2025 and Beyond

The talk "Beyond Agents: What AI Strategy Really Needs in 2025" offered a high-level perspective on the evolving AI landscape. Drawing insights from NVIDIA's GTC 2025, the speaker emphasized the convergence of AI with simulation, synthetic data, and robotics. He urged technical leaders to think beyond individual tools and embrace a more holistic approach to AI development.

The session highlighted the importance of interdisciplinary collaboration and the rise of powerful, local AI systems. The message was clear: building the future of AI requires strategic thinking, a focus on convergence, and a willingness to collaborate across domains.

## The Open Source Imperative: Building Trust and Control in AI

Finally, "The Future of AI: Building the Most Impactful Technology Together" emphasized the crucial role of open-source principles in AI development. The presenter argued that openness in language models is essential for building trust, mitigating biases, and achieving true alignment. He highlighted the growing momentum of open models and called on the community to build the next generation of AI tools collaboratively.

The core message resonated deeply: the future of AI is open. By embracing open-source principles, we can foster innovation, ensure transparency, and build AI systems that are truly beneficial to society.

## Conclusion: A Call to Action

PyConDE 2025 left me feeling inspired and energized. The talks I attended highlighted the diverse challenges and exciting opportunities in the field of AI agents. From improving data ingestion to embracing federated learning and championing open-source models, the path forward is clear: building the future of AI requires a collaborative, strategic, and ethical approach.