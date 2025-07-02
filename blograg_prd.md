# BlogRAG - Product Requirements Document (PRD)

## **1. Project Overview**

### Problem Statement
Tech professionals read numerous blog posts daily but struggle to:
- Remember specific content from articles read weeks/months ago
- Find connections between different posts they've read
- Search through their reading history effectively
- Leverage their accumulated knowledge for current work

### Solution
A personal RAG-powered knowledge base that transforms saved blog links into an intelligent, searchable system using AI embeddings.

## **2. Target Users**

**Primary User:** Tech professionals who:
- Read 5-15 technical blog posts per week
- Save links in various places (bookmarks, notion, spreadsheets)
- Need to reference past reading for current projects
- Value privacy (prefer local solutions)

**User Personas:**
- **Sarah, Senior Data Scientist**: Reads ML papers and blog posts, needs to find specific techniques she read about
- **Alex, Tech Lead**: Collects architecture articles, needs to reference patterns for design reviews
- **Jordan, Solo Developer**: Saves coding tutorials, wants to quickly find solutions to problems

## **3. Core Requirements**

### Functional Requirements

**Must Have (MVP):**
- Collect blog URLs (link, category, date_added) using a Google Chrome extension
- Extract content from blog URLs automatically
- Generate embeddings for semantic search
- Store embeddings in local vector database
- Basic search functionality with natural language queries
- Return relevant content chunks with source URLs

**Should Have (V2):**
- Web interface for search (currently CLI/script)
- Support for importing existing files (CSV, Notion export, etc.)
- Similarity scoring and ranking

**Could Have (Future):**
- Cloud deployment option
- Agentic RAG (autonomous retrieval-augmented generation workflows)

### Non-Functional Requirements

**Performance:**
- Process 100 URLs in under 10 minutes
- Search queries return results in <10 seconds
- Support up to 100 blog posts initially

**Privacy & Security:**
- All data stored locally
- User controls all data

**Usability:**
- Clear error messages for failed URL extractions
- Progress indicators for batch processing

## **4. Technical Specifications**

### Architecture
```
Chrome extention → Content Extractor → Embedder → Vector DB → RAG → Search Interface
```

### Technology Stack
- **Language:** Python
- **Dependencies:** pandas, requests, beautifulsoup4, sentence-transformers, chromadb
- **Embedding Model:** sentence-transformers (local) or OpenAI API
- **Vector DB:** ChromaDB (local, simple)
- **Package Manager:** uv

### Data Flow
1. Read Excel file with pandas (will migrate to chrome extension later)
2. For each URL: extract content, clean text, chunk
3. Generate embeddings for chunks
4. Store in ChromaDB with metadata
5. Search: embed query → similarity search → return results

### Error Handling
- Skip broken/inaccessible URLs
- Retry failed requests (3x)
- Log extraction failures
- Continue processing remaining URLs

## **5. Success Metrics**

### MVP Success Criteria
- Successfully processes 80%+ of valid blog URLs
- Search returns relevant results for test queries
- Complete end-to-end pipeline works without manual intervention
- User can find specific content they remember reading

### Quality Metrics
- Content extraction accuracy (manual spot check)
- Search relevance (user subjective evaluation)
- Processing speed (URLs per minute)
- System reliability (error rate)

## **6. Implementation Phases**

### Phase 1: Core Pipeline
- Excel file reader
- Content extractor (requests + BeautifulSoup)
- Basic embedding generation
- ChromaDB storage

### Phase 2: Search & Polish
- Search interface (CLI)
- Error handling and logging
- Testing with real data
- Documentation

### Phase 3: Enhancement
- Web interface (Streamlit or Gradio)
- Advanced search features
- Performance optimization
- User feedback integration

## **7. Risks & Mitigation**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Content extraction fails for many sites | High | Use multiple extraction methods, newspaper3k fallback |
| Embedding quality poor | Medium | Test different models, allow model switching |
| ChromaDB performance issues | Medium | Implement pagination, consider alternatives |
| User Excel format varies | Low | Flexible column mapping, validation |