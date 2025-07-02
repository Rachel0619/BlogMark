# BlogRAG - Product Requirements Document (PRD)

## **1. Project Overview**

### Problem Statement
Tech professionals read numerous blog posts daily but struggle to:
- Remember specific content from articles read weeks/months ago
- Find connections between different posts they've saved
- Search through their reading history effectively
- Leverage their accumulated knowledge for current work

### Solution
A personal RAG-powered knowledge base that transforms saved blog links into an intelligent, searchable system using AI embeddings.

## **2. Target Users**

**Primary User:** Data scientists, software engineers, tech leads who:
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
- Import blog URLs from Excel file (link, category, date_added)
- Extract content from blog URLs automatically
- Generate embeddings for semantic search
- Store embeddings in local vector database
- Basic search functionality with natural language queries
- Return relevant content chunks with source URLs

**Should Have (V2):**
- Web interface for search (currently CLI/script)
- Support for multiple file formats (CSV, Notion export)
- Content categorization and filtering
- Similarity scoring and ranking
- Export search results

**Could Have (Future):**
- Browser extension for link collection
- Multi-user support
- Cloud deployment option
- Integration with note-taking apps

### Non-Functional Requirements

**Performance:**
- Process 100 URLs in under 10 minutes
- Search queries return results in <2 seconds
- Support up to 1000 blog posts initially

**Privacy & Security:**
- All data stored locally
- No external API calls for content (embeddings API acceptable)
- User controls all data

**Usability:**
- Simple Python scripts (user knows Python)
- Clear error messages for failed URL extractions
- Progress indicators for batch processing

## **4. User Stories**

### Epic 1: Content Collection
- **As a** tech professional, **I want to** import my saved blog links from an Excel file **so that** I can process my existing collection
- **As a** user, **I want to** see progress when URLs are being processed **so that** I know the system is working
- **As a** user, **I want to** handle failed extractions gracefully **so that** one bad URL doesn't break the entire process

### Epic 2: Content Processing
- **As a** user, **I want to** extract clean text content from blog posts **so that** the search focuses on actual content, not navigation/ads
- **As a** user, **I want to** chunk long articles appropriately **so that** search results are specific and relevant
- **As a** user, **I want to** preserve metadata (category, date, source) **so that** I can filter and trace results

### Epic 3: Search & Retrieval
- **As a** user, **I want to** search using natural language **so that** I can find content without remembering exact keywords
- **As a** user, **I want to** see source URLs and dates **so that** I can access the original articles
- **As a** user, **I want to** get multiple relevant results **so that** I can compare different perspectives

## **5. Technical Specifications**

### Architecture
```
Excel File → Content Extractor → Embedder → Vector DB → Search Interface
```

### Technology Stack
- **Language:** Python (user's expertise)
- **Dependencies:** pandas, requests, beautifulsoup4, sentence-transformers, chromadb
- **Embedding Model:** sentence-transformers (local) or OpenAI API
- **Vector DB:** ChromaDB (local, simple)
- **Package Manager:** uv

### Data Flow
1. Read Excel file with pandas
2. For each URL: extract content, clean text, chunk
3. Generate embeddings for chunks
4. Store in ChromaDB with metadata
5. Search: embed query → similarity search → return results

### Error Handling
- Skip broken/inaccessible URLs
- Retry failed requests (3x)
- Log extraction failures
- Continue processing remaining URLs

## **6. Success Metrics**

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

## **7. Implementation Phases**

### Phase 1: Core Pipeline (Week 1)
- Excel file reader
- Content extractor (requests + BeautifulSoup)
- Basic embedding generation
- ChromaDB storage

### Phase 2: Search & Polish (Week 2)
- Search interface (CLI)
- Error handling and logging
- Testing with real data
- Documentation

### Phase 3: Enhancement (Week 3+)
- Web interface (Streamlit)
- Advanced search features
- Performance optimization
- User feedback integration

## **8. Risks & Mitigation**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Content extraction fails for many sites | High | Use multiple extraction methods, newspaper3k fallback |
| Embedding quality poor | Medium | Test different models, allow model switching |
| ChromaDB performance issues | Medium | Implement pagination, consider alternatives |
| User Excel format varies | Low | Flexible column mapping, validation |

## **9. Out of Scope (For MVP)**
- Real-time link collection
- Multi-user features
- Advanced NLP (summarization, tagging)
- Integration with external services
- Mobile interface

---

**Next Steps:**
1. Set up development environment with uv
2. Create project structure
3. Implement Phase 1 components
4. Test with sample data
5. Iterate based on results