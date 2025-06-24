import * as React from 'react';
import {
  Box,
  Container,
  Button,
  Paper,
  Typography,
  Tabs,
  Tab,
} from '@mui/material';
import { styled, Theme } from '@mui/material/styles';

const MessageContainer = styled(Paper)(({ theme }: { theme: Theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  maxHeight: '60vh',
  overflowY: 'auto',
}));

const PrintMessage = styled(Box)(({ theme }: { theme: Theme }) => ({
  marginBottom: theme.spacing(0.5),
  padding: theme.spacing(0.5),
  borderRadius: theme.spacing(0.5),
  backgroundColor: theme.palette.grey[50],
  fontFamily: 'monospace',
  fontSize: '0.9em',
}));

interface TraceLine {
  [key: string]: any;
}

function App() {
  const [trace, setTrace] = React.useState<TraceLine[]>([]);
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [tab, setTab] = React.useState(0);
  const [problemDescription, setProblemDescription] = React.useState<string | null>(null);
  const [generatedKernels, setGeneratedKernels] = React.useState<string[]>([]);
  const messagesEndRef = React.useRef<null | HTMLDivElement>(null);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTab(newValue);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setFileName(file.name);
      const text = await file.text();
      const lines = text.split('\n').filter((line: string) => line.trim().length > 0);
      const parsed: TraceLine[] = [];
      let foundProblem = false;
      let problem = null;
      const kernels: string[] = [];
      for (const line of lines) {
        try {
          const obj = JSON.parse(line);
          parsed.push(obj);
          // Find problem description
          if (!foundProblem && obj.type === 'prompt' && obj.data && obj.data.type === 'codegen') {
            problem = obj.data.prompt;
            foundProblem = true;
          }
          // For type 'iteration', search data.model_output and data.messages for model_output
          if (obj.type === 'iteration' && obj.data) {
            // Check for model_output directly in data
            if (obj.data.model_output !== undefined && obj.data.model_output !== null) {
              let kernel = obj.data.model_output;
              if (typeof kernel === 'object') {
                kernel = JSON.stringify(kernel, null, 2);
              }
              kernels.push(kernel);
            }
            // Check for model_output in messages
            if (Array.isArray(obj.data.messages)) {
              for (const msg of obj.data.messages) {
                if (msg.model_output !== undefined && msg.model_output !== null) {
                  let kernel = msg.model_output;
                  if (typeof kernel === 'object') {
                    kernel = JSON.stringify(kernel, null, 2);
                  }
                  kernels.push(kernel);
                }
              }
            }
          }
        } catch (e) {
          parsed.push({ error: 'Invalid JSON', raw: line });
        }
      }
      setTrace(parsed);
      setProblemDescription(problem);
      setGeneratedKernels(kernels);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Inspector
        </Typography>
        <input
          type="file"
          accept=".jsonl"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
          id="file-upload"
        />
        <label htmlFor="file-upload">
          <Button variant="contained" component="span">
            Upload .jsonl File
          </Button>
        </label>
        {fileName && (
          <Typography variant="subtitle1" sx={{ mt: 2 }}>
            Showing log for: {fileName}
          </Typography>
        )}
        <Box sx={{ mt: 2 }}>
          <Tabs value={tab} onChange={handleTabChange} aria-label="log tabs">
            <Tab label="Problem Description" />
            <Tab label="Generated Kernel" />
            <Tab label="Trace" />
          </Tabs>
          <MessageContainer>
            {tab === 0 && (
              <Box>
                {problemDescription ? (
                  <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                    {problemDescription}
                  </Typography>
                ) : (
                  <Typography variant="body2" color="textSecondary">
                    No problem description found in file.
                  </Typography>
                )}
              </Box>
            )}
            {tab === 1 && (
              <Box>
                {generatedKernels.length > 0 ? (
                  generatedKernels.map((kernel, idx) => (
                    <PrintMessage key={idx}>
                      <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                        {kernel}
                      </Typography>
                    </PrintMessage>
                  ))
                ) : (
                  <Typography variant="body2" color="textSecondary">
                    No generated kernel found in file.
                  </Typography>
                )}
              </Box>
            )}
            {tab === 2 && (
              <Box>
                {trace.length === 0 ? (
                  <Typography variant="body2" color="textSecondary">
                    No trace loaded. Please upload a .jsonl file.
                  </Typography>
                ) : (
                  trace.map((line: TraceLine, idx: number) => (
                    <PrintMessage key={idx}>
                      <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                        {JSON.stringify(line, null, 2)}
                      </Typography>
                    </PrintMessage>
                  ))
                )}
                <div ref={messagesEndRef} />
              </Box>
            )}
          </MessageContainer>
        </Box>
      </Box>
    </Container>
  );
}

export default App; 