import { loggerService } from '@logger'
import knowledgeService from '@main/services/KnowledgeService'
import { reduxService } from '@main/services/ReduxService'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import type { KnowledgeBase, KnowledgeBaseParams, KnowledgeSearchResult, Provider } from '@types'
import * as z from 'zod'

const logger = loggerService.withContext('KnowledgeServer')

const SearchArgsSchema = z.object({
  baseId: z.string().describe('Knowledge base ID'),
  query: z.string().describe('Search query string'),
  topK: z.number().optional().describe('Number of top results to return (default: 10)')
})

type McpResponse = {
  content: Array<{ type: 'text'; text: string }>
  isError?: boolean
}

async function getProviderForModel(model: { id: string; provider: string }): Promise<Provider | undefined> {
  const providers: Provider[] = await reduxService.select('state.llm.providers')
  if (!providers || !Array.isArray(providers)) return undefined
  return providers.find((p) => p.id === model.provider)
}

function resolveBaseURL(provider: Provider): string {
  let baseURL = provider.apiHost?.replace(/\/$/, '') || ''
  // Handle route endpoint '#' suffix
  if (baseURL.endsWith('#')) {
    baseURL = baseURL.slice(0, -1)
  }
  if (provider.type === 'gemini') {
    baseURL = baseURL + '/openai'
  } else if (provider.type === 'azure-openai') {
    baseURL = baseURL + '/v1'
  } else if (provider.id === 'ollama') {
    baseURL = baseURL.replace(/\/api$/, '')
  }
  return baseURL
}

async function buildKnowledgeBaseParams(base: KnowledgeBase): Promise<KnowledgeBaseParams> {
  const provider = await getProviderForModel(base.model)
  if (!provider) {
    throw new Error(`Provider '${base.model.provider}' not found for knowledge base '${base.name}'`)
  }

  const baseURL = resolveBaseURL(provider)

  return {
    id: base.id,
    dimensions: base.dimensions,
    embedApiClient: {
      model: base.model.id,
      provider: base.model.provider,
      apiKey: provider.apiKey || 'secret',
      baseURL
    },
    chunkSize: base.chunkSize,
    chunkOverlap: base.chunkOverlap,
    documentCount: base.documentCount
  }
}

class KnowledgeServer {
  public server: Server

  constructor() {
    this.server = new Server(
      {
        name: '@cherry/knowledge-server',
        version: '0.1.0'
      },
      {
        capabilities: {
          tools: {}
        }
      }
    )
    this.initialize()
  }

  initialize() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'list_knowledge_bases',
            description: 'List all available knowledge bases',
            inputSchema: {
              type: 'object',
              properties: {},
              required: []
            }
          },
          {
            name: 'search_knowledge_base',
            description: 'Search a knowledge base by ID and query string',
            inputSchema: z.toJSONSchema(SearchArgsSchema)
          }
        ]
      }
    })

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        const { name, arguments: args } = request.params
        switch (name) {
          case 'list_knowledge_bases':
            return await this.listKnowledgeBases()
          case 'search_knowledge_base': {
            const parsed = SearchArgsSchema.safeParse(args)
            if (!parsed.success) {
              throw new Error(`Invalid arguments:\n${JSON.stringify(parsed.error.format(), null, 2)}`)
            }
            return await this.searchKnowledgeBase(parsed.data.baseId, parsed.data.query, parsed.data.topK || 10)
          }
          default:
            throw new Error(`Unknown tool: ${name}`)
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error)
        return {
          content: [{ type: 'text', text: `Error: ${errorMessage}` }],
          isError: true
        }
      }
    })
  }

  private async listKnowledgeBases(): Promise<McpResponse> {
    try {
      const bases: KnowledgeBase[] = await reduxService.select('state.knowledge.bases')
      if (!bases || bases.length === 0) {
        return { content: [{ type: 'text', text: 'No knowledge bases found.' }] }
      }

      const listText = bases
        .map((b) => {
          const itemCount = b.items?.length || 0
          return `- **${b.name}** (ID: ${b.id})\n  Model: ${b.model.id} | Items: ${itemCount} | Dimensions: ${b.dimensions || 'auto'}`
        })
        .join('\n')

      return {
        content: [{ type: 'text', text: `### Knowledge Bases:\n\n${listText}` }]
      }
    } catch (error) {
      logger.error('Error listing knowledge bases:', error as Error)
      return {
        content: [
          {
            type: 'text',
            text: `Error listing knowledge bases: ${error instanceof Error ? error.message : String(error)}`
          }
        ],
        isError: true
      }
    }
  }

  private async searchKnowledgeBase(baseId: string, query: string, topK: number): Promise<McpResponse> {
    try {
      const bases: KnowledgeBase[] = await reduxService.select('state.knowledge.bases')
      const base = bases?.find((b) => b.id === baseId)
      if (!base) {
        throw new Error(`Knowledge base with ID '${baseId}' not found`)
      }

      const params = await buildKnowledgeBaseParams(base)
      const results: KnowledgeSearchResult[] = await knowledgeService.search(null as any, {
        search: query,
        base: params
      })

      const limitedResults = results.slice(0, topK)

      if (limitedResults.length === 0) {
        return { content: [{ type: 'text', text: `No results found for query: "${query}"` }] }
      }

      const resultsText = limitedResults
        .map((r, i) => {
          const source = r.metadata?.source || 'Unknown'
          return `#### ${i + 1}. (Score: ${(r.score * 100).toFixed(1)}%)\nSource: ${source}\n${r.pageContent}`
        })
        .join('\n\n')

      return {
        content: [
          {
            type: 'text',
            text: `### Search Results for "${query}":\n\nFound ${limitedResults.length} results:\n\n${resultsText}`
          }
        ]
      }
    } catch (error) {
      logger.error('Error searching knowledge base:', error as Error)
      return {
        content: [{ type: 'text', text: `Search error: ${error instanceof Error ? error.message : String(error)}` }],
        isError: true
      }
    }
  }
}

export default KnowledgeServer
