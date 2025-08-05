import os
import sys
import shutil
from pathlib import Path
from urllib.parse import urljoin
from jinja2 import Environment, FileSystemLoader

# Add parent path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities from the leaderboard module
from leaderboard.utils.export_utils import (
    get_definitions,
    grouped_definitions,
    get_leaderboard,
    get_a_definition,
    get_important_workloads
)

from leaderboard.utils.model_utils import (
    list_available_models,
    get_model_structure
)

class StaticSiteGenerator:
    def __init__(self, output_dir="./static_site", base_url="/"):
        """
        Initialize the static site generator
        
        Args:
            output_dir: Directory to output static files
            base_url: Base URL for the site (e.g., "/flashinfer-bench/" for GitHub Pages subpath)
        """
        self.output_dir = Path(output_dir)
        self.base_url = base_url
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader("leaderboard/templates"),
            autoescape=True
        )
        
        # Add custom filters/functions
        self.env.globals['url_for'] = self.url_for
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def url_for(self, endpoint, **kwargs):
        """Mock Flask's url_for function for static generation"""
        if endpoint == 'static':
            filename = kwargs.get('filename', '')
            return urljoin(self.base_url, f"static/{filename}")
        elif endpoint == 'index':
            return urljoin(self.base_url, "index.html")
        elif endpoint == 'docs':
            return urljoin(self.base_url, "docs.html")
        elif endpoint == 'models':
            return urljoin(self.base_url, "models.html")
        elif endpoint == 'model_component.show_model':
            model_name = kwargs.get('model_name', '')
            return urljoin(self.base_url, f"models/{model_name}.html")
        elif endpoint == 'leaderboard.show_leaderboard':
            definition_name = kwargs.get('definition_name', '')
            return urljoin(self.base_url, f"leaderboard/{definition_name}.html")
        return "#"
    
    def copy_static_files(self):
        """Copy static assets (CSS, JS) to output directory"""
        static_src = Path("leaderboard/static")
        static_dst = self.output_dir / "static"
        
        if static_src.exists():
            shutil.copytree(static_src, static_dst, dirs_exist_ok=True)
            print(f"✓ Copied static files to {static_dst}")
    
    def generate_index_page(self):
        """Generate the index page"""
        template = self.env.get_template("index.html")
        definitions = get_definitions()
        grouped = grouped_definitions()
        
        html = template.render(
            definitions=definitions,
            grouped=grouped,
            config={}
        )
        
        output_file = self.output_dir / "index.html"
        output_file.write_text(html)
        print(f"✓ Generated {output_file}")
    
    def generate_leaderboard_pages(self):
        """Generate individual leaderboard pages for each definition"""
        template = self.env.get_template("leaderboard.html")
        leaderboard_data = get_leaderboard()
        
        # Create leaderboard directory
        leaderboard_dir = self.output_dir / "leaderboard"
        leaderboard_dir.mkdir(exist_ok=True)
        
        for definition_name in leaderboard_data:
            try:
                definition = get_a_definition(definition_name)
                important_workloads = get_important_workloads(definition)
                
                entries = leaderboard_data[definition_name]
                
                # Process entries by device and workload
                from collections import defaultdict
                entries_by_device_and_workload = defaultdict(lambda: defaultdict(list))
                
                for device, entries_for_device in entries.items():
                    for entry in entries_for_device:
                        workload = entry["workload"]
                        entries_by_device_and_workload[device][workload].append(entry)
                
                html = template.render(
                    definition=definition,
                    entries_by_device_and_workload=dict(entries_by_device_and_workload),
                    important_workloads=important_workloads,
                    config={}
                )
                
                output_file = leaderboard_dir / f"{definition_name}.html"
                output_file.write_text(html)
                print(f"✓ Generated {output_file}")
            except Exception as e:
                print(f"✗ Failed to generate leaderboard for {definition_name}: {e}")
    
    def generate_docs_page(self):
        """Generate the documentation page"""
        template = self.env.get_template("docs.html")
        
        # Read the markdown content
        docs_file = Path("leaderboard/docs/index.md")
        docs_content = ""
        if docs_file.exists():
            import markdown
            docs_content = markdown.markdown(docs_file.read_text())
        
        html = template.render(
            content=docs_content,
            config={}
        )
        
        output_file = self.output_dir / "docs.html"
        output_file.write_text(html)
        print(f"✓ Generated {output_file}")
    
    def generate_models_page(self):
        """Generate the models list page"""
        template = self.env.get_template("models.html")
        
        try:
            models = list_available_models()
        except:
            models = []
        
        html = template.render(
            models=models,
            config={}
        )
        
        output_file = self.output_dir / "models.html"
        output_file.write_text(html)
        print(f"✓ Generated {output_file}")
    
    def generate_model_pages(self):
        """Generate individual model detail pages"""
        template = self.env.get_template("model_component.html")
        
        # Create models directory
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        try:
            models = list_available_models()
            for model_info in models:
                try:
                    model_name = model_info.get("file_name", model_info.get("model_name"))
                    model_data = get_model_structure(model_name)
                    
                    if model_data:
                        html = template.render(
                            model_name=model_data.get("model_name", model_name),
                            structure=model_data.get("structure", {}),
                            config={}
                        )
                        
                        output_file = models_dir / f"{model_name}.html"
                        output_file.write_text(html)
                        print(f"✓ Generated {output_file}")
                except Exception as e:
                    print(f"✗ Failed to generate model page for {model_name}: {e}")
        except Exception as e:
            print(f"✗ Could not generate model pages (models data not available): {e}")
    
    def generate_404_page(self):
        """Generate 404 error page"""
        template = self.env.get_template("404.html")
        html = template.render(config={})
        
        output_file = self.output_dir / "404.html"
        output_file.write_text(html)
        print(f"✓ Generated {output_file}")
    
    def generate_all(self):
        """Generate all static pages"""
        print("\n🚀 Starting static site generation...\n")
        
        # Copy static files
        self.copy_static_files()
        
        # Generate pages
        self.generate_index_page()
        self.generate_leaderboard_pages()
        self.generate_docs_page()
        self.generate_models_page()
        self.generate_model_pages()
        self.generate_404_page()
        
        print("\n✅ Static site generation complete!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate static site for FlashInfer Bench Leaderboard")
    parser.add_argument(
        "--output",
        "-o",
        default="./static_site",
        help="Output directory for static files (default: ./static_site)"
    )
    parser.add_argument(
        "--base-url",
        default="/",
        help="Base URL for the site (e.g., '/flashinfer-bench/' for GitHub Pages subpath)"
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset directory (overrides FLASHINFER_BENCH_DATASET_PATH env var)"
    )
    
    args = parser.parse_args()
    
    # Set dataset path if provided
    if args.dataset_path:
        os.environ["FLASHINFER_BENCH_DATASET_PATH"] = args.dataset_path
    
    # Generate static site
    generator = StaticSiteGenerator(
        output_dir=args.output,
        base_url=args.base_url
    )
    generator.generate_all()


if __name__ == "__main__":
    main()