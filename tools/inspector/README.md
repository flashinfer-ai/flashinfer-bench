# How to run the Inspector frontend

The Inspector is a React-based web application for viewing and analyzing kernel generation logs in JSONL format.

1️⃣ Navigate to the inspector directory:

```bash
cd inspector
```

2️⃣ Install dependencies:

```bash
npm install
```

3️⃣ Start the development server:

```bash
npm run dev
```

You should see:

```
  VITE v4.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

4️⃣ Open your browser:

```
http://localhost:3000/
```

5️⃣ Upload a `.jsonl` file containing kernel generation logs to view:
   - **Problem Description**: The original problem prompt
   - **Generated Kernel**: All generated CUDA kernels from iterations
   - **Trace**: Complete JSON log of the generation process
