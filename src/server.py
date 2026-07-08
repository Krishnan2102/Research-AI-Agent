from mcp.server.fastmcp import FastMCP
import wikipedia

# Initialize the server
mcp = FastMCP("Wikipedia_Research_Server")

@mcp.tool()
def search_wikipedia(query: str, sentences: int = 5) -> str:
    """Search Wikipedia and return a detailed summary of the matching page."""
    try:
       
        return wikipedia.summary(query, sentences=sentences)
    
    except wikipedia.exceptions.DisambiguationError as e:
       
        options = ", ".join(e.options[:5])
        return f"Query is too broad or ambiguous. Try one of these specific terms: {options}"
    
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this exact query. Try a different keyword."
    
    except Exception as e:
        return f"Error connecting to Wikipedia: {str(e)}"

if __name__ == "__main__":
    
    mcp.run(transport="sse")