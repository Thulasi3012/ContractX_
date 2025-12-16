from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder

kg_builder = KnowledgeGraphBuilder()

router = APIRouter(
    prefix="/api/Knowledge_graph",
    tags=["Knowledge Graph"]
)
@router.post("/query_knowledge_graph")
async def query_knowledge_graph(cypher_query: str = Query(..., description="Cypher query to execute")):
    """Query the knowledge graph using Cypher"""
    try:
        results = kg_builder.query_graph(cypher_query)
        return JSONResponse(content={"results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/clear_knowledge_graph")
async def clear_knowledge_graph():
    """Clear the entire knowledge graph (use with caution!)"""
    try:
        kg_builder.clear_graph()
        return JSONResponse(content={"status": "Graph cleared successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
