package swarmgo

import (
	"encoding/json"
	"fmt"

	"github.com/invopop/jsonschema"
	"github.com/prathyushnallamothu/swarmgo/llm"
)

// Agent represents an entity with specific attributes and behaviors.
type Agent struct {
	Name              string                                               // The name of the agent.
	Model             string                                               // The model identifier.
	Provider          llm.LLMProvider                                      // The LLM provider to use.
	Config            *ClientConfig                                        // Provider-specific configuration.
	Instructions      string                                               // Static instructions for the agent.
	InstructionsFunc  func(contextVariables map[string]interface{}) string // Function to generate dynamic instructions based on context.
	Functions         []AgentFunction[map[string]interface{}]              // A list of functions the agent can perform.
	Memory            *MemoryStore                                         // Memory store for the agent.
	ParallelToolCalls bool                                                 // Whether to allow parallel tool calls.
}

type AgentFunctionExecutor[I any] func(args I, contextVariables map[string]interface{}) Result

// AgentFunction represents a function that can be performed by an agent
type AgentFunction[I any] struct {
	Name        string                   // The name of the function.
	Description string                   // Description of what the function does.
	params      map[string]interface{}   // The parameters of the function.
	executor    AgentFunctionExecutor[I] // The actual function implementation.
}

// FunctionToDefinition converts an AgentFunction to a llm.Function
func FunctionToDefinition[I any](af AgentFunction[I]) llm.Function {
	return llm.Function{
		Name:        af.Name,
		Description: af.Description,
		Parameters:  af.params,
	}
}

// NewAgentFunction creates a new agent function
func NewAgentFunction[I any](name, description string, executor AgentFunctionExecutor[I]) (AgentFunction[map[string]interface{}], error) {
	var zero I
	reflector := jsonschema.Reflector{
		RequiredFromJSONSchemaTags: true,
		AllowAdditionalProperties:  false,
		DoNotReference:             true,
	}
	schema := reflector.Reflect(zero)

	// Pretty print the JSON schema
	schemaBytes, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return AgentFunction[map[string]interface{}]{}, fmt.Errorf("Error generating schema: %v", err)
	}

	var schemaMap map[string]interface{}
	if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
		return AgentFunction[map[string]interface{}]{}, fmt.Errorf("Error unmarshaling schema: %v", err)
	}

	params := make(map[string]interface{})
	for k, _ := range schemaMap {
		// Ignore JSON schema metadata
		if k[0] == '$' {
			continue
		}

		params[k] = schemaMap[k]
	}

	return AgentFunction[map[string]interface{}]{
		Name:        name,
		Description: description,
		params:      params,
		executor: func(args map[string]interface{}, contextVariables map[string]interface{}) Result {
			argsBytes, err := json.Marshal(args)
			if err != nil {
				return Result{
					Success: false,
					Error:   fmt.Errorf("error marshaling arguments: %v", err),
				}
			}

			var typedArgs I
			if err := json.Unmarshal(argsBytes, &typedArgs); err != nil {
				return Result{
					Success: false,
					Error:   fmt.Errorf("error unmarshaling arguments: %v", err),
				}
			}
			return executor(typedArgs, contextVariables)
		},
	}, nil
}

// NewAgent creates a new agent with initialized memory store
func NewAgent(
	name,
	model string, provider llm.LLMProvider) *Agent {
	return &Agent{
		Name:     name,
		Model:    model,
		Provider: provider,
		Memory:   NewMemoryStore(100), // Default to 100 short-term memories
	}
}

// WithFunctions sets the functions available to the agent
func (a *Agent) WithFunctions(functions ...AgentFunction[map[string]interface{}]) *Agent {
	a.Functions = append(a.Functions, functions...)
	return a
}

// WithConfig sets the configuration for the agent
func (a *Agent) WithConfig(config *ClientConfig) *Agent {
	a.Config = config
	return a
}

// WithInstructions sets the static instructions for the agent
func (a *Agent) WithInstructions(instructions string) *Agent {
	a.Instructions = instructions
	return a
}

// WithInstructionsFunc sets the dynamic instructions function for the agent
func (a *Agent) WithInstructionsFunc(f func(map[string]interface{}) string) *Agent {
	a.InstructionsFunc = f
	return a
}

// WithParallelToolCalls enables or disables parallel tool calls
func (a *Agent) WithParallelToolCalls(enabled bool) *Agent {
	a.ParallelToolCalls = enabled
	return a
}
