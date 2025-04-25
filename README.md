# PLN2

Este proyecto reúne las entregas de los Trabajos Prácticos 1, 2 y 3 de la materia **Procesamiento de Lenguaje Natural 2**.

A lo largo de los TPs, se construyó un sistema de agentes inteligentes basado en técnicas de **RAG (Retrieval-Augmented Generation)** y **LangGraph**.

---

## TP1 - Chatbot RAG para Documentos Personales

- **Objetivo:** Construir un chatbot capaz de indexar documentos personales (por ejemplo, un **CV en PDF**) y responder preguntas sobre su contenido.
- **Tecnologías utilizadas:**
  - Extracción de texto de PDFs
  - Embeddings y vector stores
  - RAG para recuperación y generación de respuestas
- **Resultado:** El chatbot puede responder consultas específicas sobre el contenido de un CV cargado por el usuario.

---

## TP2 - Grafo de Agentes con LangGraph

- **Objetivo:** Mejorar el sistema utilizando un **grafo de agentes** donde cada agente es responsable de un **namespace** diferente de documentos (un CV distinto por namespace).
- **Tecnologías utilizadas:**
  - LangGraph para la construcción de flujos multiagente
  - Retrieval especializado por namespace
- **Resultado:** Cada agente del grafo es capaz de hacer retrieving eficiente sobre su propio CV, discriminando correctamente los documentos.

---

## TP3 - Sistema de Agentes de Atención al Cliente para un Shopping

- **Objetivo:** Crear un **sistema de agentes** que asistan a clientes en un shopping, resolviendo tareas como:
  - Devolución de productos
  - Compras
  - Aplicación de descuentos
- **Tecnologías utilizadas:**
  - LangGraph para orquestar agentes
  - Razonamiento y generación de respuestas contextualizadas
- **Resultado:** Sistema multiagente que maneja distintos tipos de interacciones de clientes en un entorno comercial simulado.

---

## 🚀 Tecnologías generales

- Python
- LangGraph
- LangChain
- OpenAI / LLMs
- Vectorstores (Pinecone)
- Streamlit y LangGraph Studio (para visualización y pruebas)

---

## 📂 Estructura del proyecto

```
personal-chatbot/
├── rag/                      # Código del TP1
├── agentes/                  # Código del TP2
├── personal_shopper/         # Código del TP3
├── README.md                 # Este archivo
└── requirements.txt          # Dependencias del proyecto
```

---

## 🛠️ Cómo correr el proyecto

1. Clonar el repositorio.
2. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # o .venv\Scripts\activate en Windows
   ```
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar los scripts de cada TP según corresponda.

Créditos

El ejemplo de agentes de atención al cliente fue adaptado a partir del proyecto personal_shopper de OpenAI, modificándolo para utilizar LangGraph como motor de orquestación.

