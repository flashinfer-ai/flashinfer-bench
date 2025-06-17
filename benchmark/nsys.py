import subprocess
from pathlib import Path
import tempfile
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("NSYS Profiler Server")

class NSysFileProfiler:
    def __init__(self):
        # self._temp_dir = tempfile.TemporaryDirectory()
        # self.output_dir = Path(self._temp_dir.name)
        # print(f"[NSysFileProfiler] Temp output dir: {self.output_dir}")
        self.output_dir = "./nsys_output"
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def profile(self, python_file: str) -> dict:
        output_file = self.output_dir / f"{Path(python_file).stem}_profile.nsys-rep"
        print(f"[NSysFileProfiler] Output file: {output_file}")
        nsys_cmd = [
            "nsys", "profile",
            "--output", str(output_file),
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--duration", "30",  # Max 30 seconds
            "--sample", "none",  # Disable CPU sampling for faster profiling
            # "--capture-range", "cudaProfilerApi",  # Use CUDA profiler API
            "python", str(python_file)
        ]
        
        try:
            print(f"Running NSys command: {' '.join(nsys_cmd)}")
            result = subprocess.run(
                nsys_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            print(f"[NSysFileProfiler] Running command: {' '.join(nsys_cmd)}")
            print(result.stdout)
            if result.returncode == 0:
                return {
                    "success": True,
                    "output_file": str(output_file),
                    "stdout": result.stdout
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "output_file": str(output_file)
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "NSys timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_nvtx_info(self, nsys_file: str) -> dict:
        try:
            cmd = [
                "nsys", "stats", 
                "--report", "nvtx_sum",
                "--format", "csv",
                "--force-export", "true",
                str(nsys_file)
            ]
            print(f"[NSysFileProfiler] Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            lines = result.stdout.strip().split('\n')
            header = lines[2].split(',')
            duration_idx = header.index("Total Time (ns)")
            name_idx = header.index("Range")
            info = []
            for line in lines[3:]:
                fields = line.split(',')
                duration_ns = float(fields[duration_idx].strip('"'))
                name = fields[name_idx].strip('"')
                info.append({"name": name, "duration_ms": duration_ns / 1e6})
            return {"success": True, "nvtx": info}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "NSys stats NVTX timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_kernel_times(self, nsys_file: str) -> dict:
        try:
            cmd = ["nsys", "stats", "--report", "gputrace", "--format", "csv", "--force-export", "true", nsys_file]
            print(f"[NSysFileProfiler] Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {"success": False, "error": result.stderr}
            lines = result.stdout.strip().split('\n')
            header = lines[2].split(',')
            duration_idx = header.index("Duration (ns)")
            name_idx = header.index("Name")
            total_ns = 0
            count = 0
            for line in lines[3:]:
                fields = line.split(',')
                duration_ns = float(fields[duration_idx].strip('"'))
                name = fields[name_idx].strip('"').lower()
                if not any(x in name for x in ['memcpy', 'memset']):
                    total_ns += duration_ns
                    count += 1
            return {
                "success": True,
                "kernel_time_us": total_ns / 1e3,
                "kernel_count": count
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "NSys stats gputrace timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

profiler = NSysFileProfiler()

@mcp.tool()
def nsys_profile_file(python_file: str) -> dict:
    """Run NSys profile on a Python file."""
    return profiler.profile(python_file)

@mcp.tool()
def nsys_parse_nvtx(nsys_file: str) -> dict:
    """Extract NVTX info from an NSYS file."""
    return profiler.extract_nvtx_info(nsys_file)

@mcp.tool()
def nsys_parse_kernel_times(nsys_file: str) -> dict:
    """Extract kernel execution times from an NSYS file."""
    return profiler.extract_kernel_times(nsys_file)

if __name__ == "__main__":
    mcp.run(transport='stdio')