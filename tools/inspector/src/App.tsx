import * as React from 'react';
import {
  Box,
  Container,
  Button,
  Paper,
  Typography,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  List
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled, Theme } from '@mui/material/styles';

const MessageContainer = styled(Paper)(({ theme }: { theme: Theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  maxHeight: '60vh',
  overflowY: 'auto',
}));

interface TraceLine {
  [key: string]: any;
}

function AgentInspectorTraceItem({ line, idx }: { line: TraceLine; idx: number }) {
  const type = line.type;
  // Helper to render a multi-line text field
  const renderTextField = (label: string, value: string | undefined) => (
    <TextField
      label={label}
      value={value || ''}
      fullWidth
      multiline
      minRows={2}
      InputProps={{ readOnly: true }}
      margin="dense"
      variant="outlined"
      sx={{ mb: 1 }}
    />
  );

  if (type === 'llm_request') {
    const modelName = line.model_name || line.model || (line.data && line.data.model_name);
    const conversation = line.conversation || (line.data && line.data.conversation);
    const llmOutput = line.llm_output || (line.data && line.data.llm_output);
    const toolCalls = line.tool_calls || (line.data && line.data.tool_calls);
    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">llm_request #{idx + 1}</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {renderTextField('Model Name', modelName)}
          {conversation && Array.isArray(conversation) && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>Conversation:</Typography>
              <List dense>
                {conversation.map((msg: any, i: number) => (
                  <Accordion key={i} sx={{ mb: 1 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="body2">Message #{i + 1}</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {Object.entries(msg).map(([k, v]) =>
                        typeof v === 'string' ? renderTextField(k, v) : (
                          <Box key={k} sx={{ mb: 1 }}>
                            <Typography variant="caption">{k}:</Typography>
                            <pre style={{ margin: 0 }}>{JSON.stringify(v, null, 2)}</pre>
                          </Box>
                        )
                      )}
                    </AccordionDetails>
                  </Accordion>
                ))}
              </List>
            </Box>
          )}
          {renderTextField('LLM Output', llmOutput)}
          {toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0 && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>Tool Calls:</Typography>
              <List dense>
                {toolCalls.map((tool: any, i: number) => (
                  <Accordion key={i} sx={{ mb: 1 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="body2">Tool Call #{i + 1}</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {renderTextField('Tool Name', tool.tool_name)}
                      {tool.arguments && (
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="caption">Arguments:</Typography>
                          <pre style={{ margin: 0 }}>{JSON.stringify(tool.arguments, null, 2)}</pre>
                        </Box>
                      )}
                    </AccordionDetails>
                  </Accordion>
                ))}
              </List>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>
    );
  } else if (type === 'tool_call' || type === 'mcp') {
    // Show all fields in the schema
    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">{type} #{idx + 1}</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {Object.entries(line).map(([k, v]) =>
            typeof v === 'string' ? renderTextField(k, v) : (
              <Box key={k} sx={{ mb: 1 }}>
                <Typography variant="caption">{k}:</Typography>
                <pre style={{ margin: 0 }}>{JSON.stringify(v, null, 2)}</pre>
              </Box>
            )
          )}
        </AccordionDetails>
      </Accordion>
    );
  } else {
    // Fallback: show as JSON
    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle2">Unknown type #{idx + 1}</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <pre style={{ margin: 0 }}>{JSON.stringify(line, null, 2)}</pre>
        </AccordionDetails>
      </Accordion>
    );
  }
}

function renderJsonField(label: string, value: any) {
  if (typeof value === 'string') {
    return (
      <TextField
        label={label}
        value={value}
        fullWidth
        multiline
        minRows={2}
        InputProps={{ readOnly: true }}
        margin="dense"
        variant="outlined"
        sx={{ mb: 1 }}
      />
    );
  } else if (typeof value === 'object' && value !== null) {
    return (
      <Accordion sx={{ mb: 1 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2">{label}</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <pre style={{ margin: 0 }}>{JSON.stringify(value, null, 2)}</pre>
        </AccordionDetails>
      </Accordion>
    );
  } else {
    return (
      <TextField
        label={label}
        value={String(value)}
        fullWidth
        InputProps={{ readOnly: true }}
        margin="dense"
        variant="outlined"
        sx={{ mb: 1 }}
      />
    );
  }
}

function renderDefinition(def: any) {
  if (!def) return null;
  return (
    <Box>
      {renderJsonField('Name', def.name)}
      {renderJsonField('Type', def.type)}
      {renderJsonField('Description', def.description)}
      {renderJsonField('Axes', def.axes)}
      {renderJsonField('Inputs', def.inputs)}
      {renderJsonField('Outputs', def.outputs)}
      {renderJsonField('Reference', def.reference)}
      {renderJsonField('Constraints', def.constraints)}
    </Box>
  );
}

function renderSolution(sol: any) {
  if (!sol) return null;
  return (
    <Box>
      {renderJsonField('Name', sol.name)}
      {renderJsonField('Definition', sol.definition)}
      {renderJsonField('Description', sol.description)}
      {renderJsonField('Author', sol.author)}
      {renderJsonField('Spec', sol.spec)}
      {sol.sources && Array.isArray(sol.sources) && (
        <Accordion sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="body2">Sources</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {sol.sources.map((src: any, i: number) => (
              <Accordion key={i} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="body2">{src.path}</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TextField
                    label="Content"
                    value={src.content}
                    fullWidth
                    multiline
                    minRows={4}
                    InputProps={{ readOnly: true }}
                    margin="dense"
                    variant="outlined"
                  />
                </AccordionDetails>
              </Accordion>
            ))}
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
}

function renderTrace(trace: any) {
  if (!trace) return null;
  // Top-level fields: definition, solution, workload, evaluation
  return (
    <Box>
      {renderJsonField('Definition', trace.definition)}
      {renderJsonField('Solution', trace.solution)}
      {renderJsonField('Workload', trace.workload)}
      {renderJsonField('Evaluation', trace.evaluation)}
    </Box>
  );
}

function App() {
  const [tab, setTab] = React.useState(0);
  // State for each file type
  const [definition, setDefinition] = React.useState<any>(null);
  const [definitionFile, setDefinitionFile] = React.useState<string | null>(null);
  const [solution, setSolution] = React.useState<any>(null);
  const [solutionFile, setSolutionFile] = React.useState<string | null>(null);
  const [trace, setTrace] = React.useState<any>(null);
  const [traceFile, setTraceFile] = React.useState<string | null>(null);
  // For agent inspector (trace elements)
  const [traceElements, setTraceElements] = React.useState<any[]>([]);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTab(newValue);
  };

  // File upload handlers
  const handleDefinitionUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setDefinitionFile(file.name);
      const text = await file.text();
      try {
        setDefinition(JSON.parse(text));
      } catch {
        setDefinition(null);
      }
    }
  };
  const handleSolutionUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSolutionFile(file.name);
      const text = await file.text();
      try {
        setSolution(JSON.parse(text));
      } catch {
        setSolution(null);
      }
    }
  };
  const handleTraceUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setTraceFile(file.name);
      const text = await file.text();
      try {
        const obj = JSON.parse(text);
        setTrace(obj);
        // For agent inspector: if obj has a 'trace' or 'elements' array, use it; else, fallback to []
        if (Array.isArray(obj.trace)) {
          setTraceElements(obj.trace);
        } else if (Array.isArray(obj.elements)) {
          setTraceElements(obj.elements);
        } else if (Array.isArray(obj)) {
          setTraceElements(obj);
        } else {
          setTraceElements([]);
        }
      } catch {
        setTrace(null);
        setTraceElements([]);
      }
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Inspector
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Tabs value={tab} onChange={handleTabChange} aria-label="schema tabs">
            <Tab label="Definition" />
            <Tab label="Solution" />
            <Tab label="Trace" />
          </Tabs>
          <MessageContainer>
            {tab === 0 && (
              <Box>
                <input
                  type="file"
                  accept=".json,.jsonl"
                  onChange={handleDefinitionUpload}
                  style={{ display: 'none' }}
                  id="definition-upload"
                />
                <label htmlFor="definition-upload">
                  <Button variant="contained" component="span">
                    Upload Definition File
                  </Button>
                </label>
                {definitionFile && (
                  <Typography variant="subtitle1" sx={{ mt: 2 }}>
                    Showing: {definitionFile}
                  </Typography>
                )}
                {definition && renderDefinition(definition)}
              </Box>
            )}
            {tab === 1 && (
              <Box>
                <input
                  type="file"
                  accept=".json,.jsonl"
                  onChange={handleSolutionUpload}
                  style={{ display: 'none' }}
                  id="solution-upload"
                />
                <label htmlFor="solution-upload">
                  <Button variant="contained" component="span">
                    Upload Solution File
                  </Button>
                </label>
                {solutionFile && (
                  <Typography variant="subtitle1" sx={{ mt: 2 }}>
                    Showing: {solutionFile}
                  </Typography>
                )}
                {solution && renderSolution(solution)}
              </Box>
            )}
            {tab === 2 && (
              <Box>
                <input
                  type="file"
                  accept=".json,.jsonl"
                  onChange={handleTraceUpload}
                  style={{ display: 'none' }}
                  id="trace-upload"
                />
                <label htmlFor="trace-upload">
                  <Button variant="contained" component="span">
                    Upload Trace File
                  </Button>
                </label>
                {traceFile && (
                  <Typography variant="subtitle1" sx={{ mt: 2 }}>
                    Showing: {traceFile}
                  </Typography>
                )}
                {trace && renderTrace(trace)}
                {/* Agent Inspector: show all trace elements in a list, each in a drop-down box */}
                {traceElements.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" sx={{ mb: 1 }}>Agent Inspector</Typography>
                    {traceElements.map((line, idx) => (
                      <AgentInspectorTraceItem key={idx} line={line} idx={idx} />
                    ))}
                  </Box>
                )}
              </Box>
            )}
          </MessageContainer>
        </Box>
      </Box>
    </Container>
  );
}

export default App; 