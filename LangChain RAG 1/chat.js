import "dotenv/config";

import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings, ChatOllama } from "@langchain/ollama";

async function main() {
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: process.env.OLLAMA_BASE_URL,
  });

  const vectorStore = new Chroma(embeddings, {
    collectionName: "pdf-rag",
    host: "localhost",
    port: 8000,
  });

  const model = new ChatOllama({
    model: "gemma4:latest",
    baseUrl: process.env.OLLAMA_BASE_URL,
  });

  const rl = readline.createInterface({ input, output });

  while (true) {
    const question = await rl.question("\nAsk a question (or 'exit'): ");

    if (question.toLowerCase() === "exit") break;

    const docs = await vectorStore.similaritySearch(question, 4);

    console.log("\nRetrieved Chunks:\n");

    docs.forEach((doc, index) => {
      console.log(`Chunk ${index + 1}`);
      console.log(doc.pageContent);
      console.log("------------------------------");
    });

    const context = docs.map((d) => d.pageContent).join("\n\n");

    const response = await model.invoke(`
        Answer ONLY using the context below.

        Context:
        ${context}

        Question:
        ${question}
        `);

    console.log("\nAnswer:\n");
    console.log(response.content);
  }

  rl.close();
}

main();
