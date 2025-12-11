---
draft: false 
date: 2025-12-11
authors: 
 - sonam
slug: Year-in-review-2025
title: Beyond EmbedAnything!
---
# Beyond EmbedAnything: **A Year of Growth Beyond Expectations**!!

**Reflecting on EmbedAnything: A Year Later**

A year ago, I shared the story behind EmbedAnything—how we built and scaled an embedding infrastructure that has since been loved by developers at Microsoft, Meta, Tencent, AWS, ByteDance and RedHat. In that post, I documented our journey: the technical decisions that enabled massive scale, the enterprise collaborations that shaped our direction, and our unwavering commitment to building best-in-class infrastructure for RAG and agentic systems.

<!-- more -->


![IMG_20251031_212241.jpg](https://royal-hygienic-522.notion.site/image/attachment%3A8582a448-6c26-4777-9f95-45e7d9adb22a%3AIMG_20251031_212241.jpg?table=block&id=2c581b6a-6bbe-8020-8c48-d6685bf6dcbb&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1420&userId=&cache=v2)

I also made a point to be transparent about our mistakes alongside our successes. Building great products requires learning from what doesn't work as much as celebrating what does. If you're interested in the technical architecture, scaling challenges, or the candid lessons we learned along the way, [https://embed-anything.com/blog/2024/12/15/embed-anything/](https://www.notion.so/Beyond-EmbedAnything-A-Year-of-Growth-Beyond-Expectations-2c481b6a6bbe8015a550dd7b5504626c?pvs=21).

## A Year of Growth Beyond Expectations

This past year exceeded my expectations in ways I hadn't anticipated. 

**By the numbers:** We grew 3X, and our GitHub star count increased from 200 to 870+. But the metrics that matter most can't be captured in downloads alone.

**What truly stands out** is the organic community that's emerged around EmbedAnything. Developers are actively discussing the project on social media and at conferences, leading to meaningful contributions and technical discussions on our GitHub repository. This kind of engagement is what makes open-source work rewarding.

<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://royal-hygienic-522.notion.site/image/attachment%3A4008ab3f-852e-4418-8f54-51d5a83fb17f%3Aimage.png?table=block&id=2c581b6a-6bbe-80b8-b4dc-daf38230c191&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=660&userId=&cache=v2" alt="image.png" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="https://royal-hygienic-522.notion.site/image/attachment%3Afbbde121-f682-4c79-99b4-1b7ceb93b842%3Aimage.png?table=block&id=2c581b6a-6bbe-801b-9db5-cc7d5c7b7f09&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=2000&userId=&cache=v2" alt="image.png" style="flex: 1; max-width: 50%; height: auto;" />
</div>

![image.png](https://royal-hygienic-522.notion.site/image/attachment%3A698c0698-d93a-4be1-bfe2-b7de732f8b08%3Aimage.png?table=block&id=2c581b6a-6bbe-80eb-b50a-f8c6d0a5bdf9&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=660&userId=&cache=v2)

<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://royal-hygienic-522.notion.site/image/attachment%3A22ceb3e8-922c-46af-af1b-d4e1de445bfa%3Aimage.png?table=block&id=2c581b6a-6bbe-8030-b503-dcd73f7c0565&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=660&userId=&cache=v2" alt="image.png" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="https://royal-hygienic-522.notion.site/image/attachment%3A0cd544ba-bff3-433f-914c-66f966dabe03%3Aimage.png?table=block&id=2c581b6a-6bbe-80a8-847a-d3cec6a3d4b3&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=660&userId=&cache=v2" alt="image.png" style="flex: 1; max-width: 50%; height: auto;" />
</div>

I've been fortunate to witness this enthusiasm firsthand. Over the past year, I've had the opportunity to present EmbedAnything at Berlin Buzzwords, PyCon DE, GDG Berlin, and DevFest

I've also shared our work in the offices of Google, Deutsche Bank, JetBrains, and Zed, engaging directly with teams using similar technologies and facing comparable challenges.

These conversations—whether at conferences, in office visits, or through GitHub issues—have been invaluable in shaping our roadmap and understanding real-world use cases.

## **Some of the contributions I want to highlight**

1. Jack Bosewell: Processor Crate:  which takes files of different types and produces metadata-rich descriptions, extremely useful for RAG.
2. Taradeepan’s AWS: Now you can embed a file directly from your AWS S3 bucket
3.  Contributions from Milvus, Qdrant, and SingleStore: Making us support almost all the vector databases as adapters for vector streaming, making as the standard Infrastructure for RAG.

Others, like adding HF home, endpoint, fixing Qwen instruction prompt, fixing versions, etc



## **Project highlights**

I am only mentioning a few of them that I really liked
1. FogX-Store is a dataset store service that collects and serves large robotics datasets: https://github.com/J-HowHuang/FogX-Store \n
2. A Rust-based cursor like chat with your codebase tool: https://github.com/timpratim/cargo-chat \n
3. Semantic file tracker in CLI operated through a daemon built with rust.: https://github.com/sam-salehi/sophist \n

Find the full list here: https://github.com/StarlightSearch/EmbedAnything?tab=readme-ov-file#awesome-projects-built-on-embedanything

## **Things that we did right:**

1. **StarlightSearch Discussions**, community feedback has always been our compass. It's what enabled us to build solutions that truly differentiate us in the market. Our GitHub discussions are a testament to this—they reflect real conversations with real developers solving real problems, and they directly inform our roadmap.
    

    

1. **Becoming the Standard Infrastructure for RAG with vector Streaming**. One of the most validating aspects of EmbedAnything's growth has been the collaboration with the vector database ecosystem. Nearly every major vector database provider has contributed adapters to our library, giving developers unprecedented modularity in their tech stack choices.

![image.png](https://royal-hygienic-522.notion.site/image/attachment%3Ae14eb937-88a7-4024-8a5d-9717b8e5b992%3Aimage.png?table=block&id=2c581b6a-6bbe-80d5-b51c-c354786892f9&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1420&userId=&cache=v2)

Milvus even published a dedicated blog post about EmbedAnything, highlighting how our library integrates with their platform. This kind of recognition from established players in the space validates both our technical approach and the value we're providing to the developer community.

**Expanding into Agents and Reinforcement Learning**

This year, you asked for agentic capabilities, and we delivered—built in Rust for performance and reliability. As the Model Context Protocol (MCP) gained traction and context engineering became increasingly critical, we focused our efforts on two key areas:

1. **CodeAct implementation** for more sophisticated agent behaviors
2. **Reinforcement learning training** for domain-specific agents, including self-improvement capabilities based on SearchR1 architectures

**First Real-World Validation: Our First On-Premise Deployment**

This year, we deployed our search product on-premises at the Serpentine, student tech group at TU/e to test scalability and real-time response capabilities under production conditions.

The results exceeded our expectations. Seeing StarlightSearch perform in a live environment—handling real queries, scaling to actual demand, and delivering answers in real-time—validated everything we've built.

What we've created isn't just technically sound on paper. It works, it scales, and it delivers results that feel transformative in practice.

This deployment gave us invaluable insights into production requirements and confirmed that our architecture can handle enterprise-scale workloads. More importantly, it demonstrated that the performance characteristics we've prioritized—speed, accuracy, and reliability—translate directly into user value.

**Writing More Technical Contents**

After every talk I presented, I found out some knowledge gap, that still exists in people's mind and wrote a technical blog about it. Read our blogs on memory leak, configuring textembedconfig for embedanything, ColPali and Fusion DeepSearch --> [Here](https://embed-anything.com/blog/).

And Subscribe to our [newletter](https://preview.mailerlite.io/forms/1975233/173424452135028700/share) to get notification for upcoming blogs on Agents and Reinformcement Learning.


<div style="display: flex; gap: 10px; align-items: center;">
  <img src="https://royal-hygienic-522.notion.site/image/attachment%3A57e9d863-7534-4b1e-b57b-af017bcb318a%3Aimage.png?table=block&id=2c581b6a-6bbe-80b9-913b-cc7fb734e1b3&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=660&userId=&cache=v2" alt="image.png" style="flex: 1; max-width: 50%; height: auto;" />
  <img src="https://royal-hygienic-522.notion.site/image/attachment%3A7fb11489-0dc5-4619-b1ca-9d98a9db1d5b%3Aimage.png?table=block&id=2c581b6a-6bbe-8032-acc2-d5a88d448e36&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=660&userId=&cache=v2" alt="image.png" style="flex: 1; max-width: 50%; height: auto;" />
</div>


## **A Difficult Decision: Closing Lumo**

We made the difficult decision to close-source Lumo, our agentic AI framework. This decision wasn't made lightly, and I want to be transparent about why.

EmbedAnything has been extensively copied by well-funded companies—organizations with substantial resources but limited interest in genuine innovation or community engagement. Some have replicated not just technical concepts, but marketing materials verbatim. While I'm choosing not to name them now, I'm documenting these instances(unethical practices) and may address them directly in the future.

**Why This Matters**

The open-source ecosystem thrives on good-faith collaboration and attribution. When companies with significant funding simply replicate rather than innovate, it creates a challenging environment for original creators operating with minimal resources (our entire investment in EmbedAnything remains €30 in stickers).

**Moving Forward**

This situation has been challenging, but it hasn't stopped our momentum. We remain committed to:

- Building differentiated products that solve real problems
- Maintaining transparency with our community
- Continuing to listen and respond to developer needs

I'm asking the open-source community to support original projects and creators. creativity, empathy, and genuine community engagement can't be purchased or copied. The organisation who copied us will always remain a second-class copy of embedanything.

## **Mistakes and learnings.**

My biggest mistake this year was delaying our sales efforts. Without the cushion of venture funding, we need revenue to sustain our open-source work—it's that simple. While we entered the agentic AI market later than some competitors, we're strategically positioned in reinforcement learning, where we believe the real differentiation lies.

**The Rust Advantage—and Its Challenges**

Building in Rust has given us significant advantages, particularly the ability to compile for any hardware architecture. This opens multiple market opportunities: consumer edge applications, robotics, and physical AI systems.

However, each market presents unique challenges:

- **Consumer applications** require substantial investment in habit formation and user education. The common objection—"Why not just use OpenAI?"—misses the point about cost optimization and data privacy, but overcoming that perception requires resources we're currently building toward.
- **Robotics and physical AI** are compelling, but require deep vertical focus.

As the saying goes: when you build for everyone, you build for no one. We're focused on identifying our core audience—developers and organizations that prioritize high-performance infrastructure and understand its value. Not every team needs what we've built, and that's perfectly fine. Our goal is to serve those who do exceptionally well.

**The Incorporation Timing Mistake**

I delayed incorporating StarlightSearch, waiting to secure our first customer before making it official. In retrospect, this was backward thinking. When you're gaining significant community traction and industry attention, incorporation provides credibility and trust that can actually accelerate customer acquisition.

## **Moving Forward: StarlightSearch is Ready**

I'm pleased to announce that **StarlightSearch is in the process of incorporation** and ready to serve high-performance Infrastructure.

I'm driven by a genuine passion for building exceptional products, advancing AI capabilities, and embracing emerging technologies. While I may not have a traditional business school background or claim any particular title like a product “leader”, my approach is grounded in a few core principles:

![image.png](https://royal-hygienic-522.notion.site/image/attachment%3A608eb6bd-0aa4-4797-a273-3382ba59594e%3Aimage.png?table=block&id=2c581b6a-6bbe-80a1-9a92-d121e28f19ce&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=880&userId=&cache=v2)

**Transparency**: I believe in open communication about our direction, decisions, and even our mistakes. You'll always know where we stand.

**Collaboration**: Your input matters. I'm committed to actively listening to your needs, feedback, and ideas. Collaborate with enterprises, small or large, and have their inputs reach to you.

**Excellence**: Our products will reflect our dedication to quality and innovation. This isn't just a promise—it's a standard I hold myself accountable to every day.

I'm undeterred in this mission and focused on what truly matters: delivering value and building something meaningful together.

**What do you need from us?** I'm listening…