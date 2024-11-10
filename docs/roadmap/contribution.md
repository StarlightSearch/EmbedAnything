# Contribution Guidelines

## 🚀 Getting Started
To get started, check the [Issues Section] for tasks labeled "Good First Issue" or "Help Needed". These issues are perfect for new contributors or those looking to make a valuable impact quickly.

If you find an issue you want to tackle:

Comment on the issue to let us know you’d like to work on it.
Wait for confirmation—an admin will assign the issue to you.
💻 Setting Up Your Development Environment
To start working on the project, follow these steps:

1. Fork the Repository: Begin by forking the repository from the dev branch. We do not allow direct contributions to the main branch.</b>
2. Clone Your Fork: After forking, clone the repository to your local machine.</b>
3. Create a New Branch: For each contribution, create a new branch following the naming convention: feature/your-feature-name or bugfix/your-bug-name.</b>

## 🛠️ Contributing Guidelines
🔍 Reporting Bugs
If you find a bug, here’s how to report it effectively:

Title: Use a clear and descriptive title, with appropriate labels.
Description: Provide a detailed description of the issue, including:
Steps to reproduce the problem.
Expected and actual behavior.</b>

Any relevant logs, screenshots, or additional context.
Submit the Bug Report: Open a new issue in the [Issues Section] and include all the details. This helps us understand and resolve the problem faster.

## 🐍 Contributing to Python Code
If you're contributing to the Python codebase, follow these steps:

1. Create an Independent File: Write your code in a new file within the python folder. </b>
2. Build with Maturin: After writing your code, use maturin build to build the package. </b>
3. Import and Call the Function: 
4. Use the following import syntax:
from embed_anything.<Library_name> import * </b>
5. Then, call the function using:
from embed_anything import <function_name> </b>
Feel free to open an issue if you encounter any problems during the process.

🧩 Contributing to Adapters
To contribute to adapters, follow these guidelines:

1. Implement Adapter Class: Create an Adapter class that supports the create, add, and delete operations for your specific use case. </b>
2. Check Existing Adapters: Use the existing Pinecone and Weaviate adapters as references to maintain consistency in structure and functionality. </b>
3. Testing: Ensure your adapter is tested thoroughly before submitting a pull request.


### 🔄 Submitting a Pull Request </b>
Once your contribution is ready: </b>

Push Your Branch: Push your branch to your forked repository.</b>

Submit a Pull Request (PR): Open a PR from your branch to the dev branch of the main repository. Ensure your PR includes:</b>

1. A clear description of the changes.</b>
2. Any relevant issue numbers (e.g., "Closes #123").</b>
3. Wait for Review: A maintainer will review your PR. Please be responsive to any feedback or requested changes.