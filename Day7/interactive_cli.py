"""
Comprehensive CLI Client for the RAG API.

A professional command-line interface providing full access to all RAG API endpoints
with an intuitive menu system, configuration management, and advanced features.
"""

import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

class RAGAPIClient:
    """Professional CLI client for the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Configuration
        self.config = {
            "default_top_k": 5,
            "default_min_similarity": 0.1,
            "show_urls": True,
            "show_timing": True,
            "max_preview_length": 150
        }
        
        # Session state
        self.query_history = []
        self.last_results = None
        
    def make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to the API with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                print(f"❌ Unsupported method: {method}")
                return None
                
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to server at {self.base_url}")
            print("   Make sure the server is running with: python src/main.py")
            return None
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def display_header(self, title: str):
        """Display a formatted section header."""
        print(f"\n{'='*60}")
        print(f"🔧 {title}")
        print(f"{'='*60}")
    
    def display_menu(self, title: str, options: list):
        """Display a menu with options."""
        print(f"\n📋 {title}")
        print("-" * 40)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        print(f"  0. 🔙 Back/Exit")
    
    def get_user_choice(self, max_option: int) -> int:
        """Get user menu choice with validation."""
        while True:
            try:
                choice = input(f"\n👉 Enter choice (0-{max_option}): ").strip()
                choice = int(choice)
                if 0 <= choice <= max_option:
                    return choice
                print(f"❌ Please enter a number between 0 and {max_option}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return 0

                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return 0

    # =========================
    # 🏠 SERVER STATUS METHODS
    # =========================
    
    def server_status_menu(self):
        """Server status and health management."""
        while True:
            self.display_menu("Server Status & Health", [
                "🏥 Health Check",
                "📊 Pipeline Status", 
                "🔌 Connection Test",
                "📈 Server Information"
            ])
            
            choice = self.get_user_choice(4)
            if choice == 0:
                break
            elif choice == 1:
                self.health_check()
            elif choice == 2:
                self.pipeline_status()
            elif choice == 3:
                self.connection_test()
            elif choice == 4:
                self.server_info()
    
    def health_check(self):
        """Check server health."""
        self.display_header("Health Check")
        result = self.make_request("/health")
        
        if result:
            status = result.get('status', 'unknown')
            message = result.get('message', 'No message')
            
            status_emoji = {
                'healthy': '✅',
                'initializing': '🔄', 
                'first_time_init': '🆕',
                'unhealthy': '❌',
                'error': '💥'
            }.get(status, '❓')
            
            print(f"{status_emoji} Status: {status.upper()}")
            print(f"💬 Message: {message}")
            
            if 'details' in result:
                details = result['details']
                print(f"\n📋 Details:")
                for key, value in details.items():
                    print(f"   {key}: {value}")
    
    def pipeline_status(self):
        """Get detailed pipeline status."""
        self.display_header("Pipeline Status")
        result = self.make_request("/status")
        
        if result:
            print(f"🔧 Overall Status: {result.get('status', 'unknown').upper()}")
            
            if 'pipeline' in result and result['pipeline']:
                pipeline = result['pipeline']
                print(f"\n📊 Pipeline Information:")
                print(f"   📚 Documents: {pipeline.get('document_count', 'N/A')}")
                print(f"   🤖 Model: {pipeline.get('model_name', 'N/A')}")
                print(f"   📁 Data Directory: {pipeline.get('data_directory', 'N/A')}")
                print(f"   ✅ Initialized: {pipeline.get('initialized', False)}")
            
            if 'api' in result:
                api_info = result['api']
                print(f"\n🌐 API Information:")
                print(f"   📦 Version: {api_info.get('version', 'N/A')}")
                if 'endpoints' in api_info:
                    print(f"   🔗 Endpoints: {', '.join(api_info['endpoints'])}")
    
    def connection_test(self):
        """Test basic connectivity."""
        self.display_header("Connection Test")
        print(f"🔌 Testing connection to: {self.base_url}")
        
        start_time = time.time()
        result = self.make_request("/")
        elapsed = (time.time() - start_time) * 1000
        
        if result:
            print(f"✅ Connection successful! ({elapsed:.1f}ms)")
            print(f"🏠 Server: {result.get('message', 'N/A')}")
            print(f"📦 Version: {result.get('version', 'N/A')}")
        else:
            print(f"❌ Connection failed after {elapsed:.1f}ms")
    
    def server_info(self):
        """Display comprehensive server information."""
        self.display_header("Server Information")
        
        # Get root info
        root_info = self.make_request("/")
        health_info = self.make_request("/health") 
        status_info = self.make_request("/status")
        
        if root_info:
            print(f"🏠 Server: {root_info.get('message', 'N/A')}")
            print(f"📦 Version: {root_info.get('version', 'N/A')}")
            print(f"🌐 Base URL: {self.base_url}")
        
        if health_info:
            print(f"🏥 Health: {health_info.get('status', 'unknown').upper()}")
            
        if status_info and 'pipeline' in status_info:
            pipeline = status_info['pipeline']
            if pipeline:
                print(f"📚 Documents: {pipeline.get('document_count', 'N/A')}")
                print(f"🤖 Model: {pipeline.get('model_name', 'N/A')}")

    # =========================
    # 🔍 SEARCH METHODS  
    # =========================
    
    def search_menu(self):
        """Search and query management."""
        while True:
            self.display_menu("Interactive Search", [
                "🔍 New Search Query",
                "📋 Quick Search (defaults)",
                "📚 Search History",
                "💾 Export Last Results",
                "🎯 Batch Search"
            ])
            
            choice = self.get_user_choice(5)
            if choice == 0:
                break
            elif choice == 1:
                self.new_search()
            elif choice == 2:
                self.quick_search()
            elif choice == 3:
                self.search_history()
            elif choice == 4:
                self.export_results()
            elif choice == 5:
                self.batch_search()
    
    def new_search(self):
        """Perform a new search with custom parameters."""
        self.display_header("New Search Query")
        
        try:
            query = input("🔍 Enter your query: ").strip()
            if not query:
                print("❌ Query cannot be empty")
                return
                
            top_k = input(f"📊 Top K results (default {self.config['default_top_k']}): ").strip()
            top_k = int(top_k) if top_k else self.config['default_top_k']
            
            min_sim = input(f"📈 Min similarity (default {self.config['default_min_similarity']}): ").strip()
            min_sim = float(min_sim) if min_sim else self.config['default_min_similarity']
            
            self.perform_search(query, top_k, min_sim)
            
        except ValueError:
            print("❌ Invalid parameter values")
        except KeyboardInterrupt:
            print("\n🚫 Search cancelled")
    
    def quick_search(self):
        """Quick search with default parameters."""
        self.display_header("Quick Search")
        
        try:
            query = input("🔍 Enter your query: ").strip()
            if not query:
                print("❌ Query cannot be empty")
                return
                
            self.perform_search(query, self.config['default_top_k'], self.config['default_min_similarity'])
            
        except KeyboardInterrupt:
            print("\n🚫 Search cancelled")
    
    def perform_search(self, query: str, top_k: int, min_similarity: float):
        """Execute a search and display results."""
        data = {
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity
        }
        
        print(f"\n🔍 Searching for: '{query}'")
        print(f"📊 Parameters: top_k={top_k}, min_similarity={min_similarity}")
        
        start_time = time.time()
        result = self.make_request("/search", "POST", data)
        client_time = (time.time() - start_time) * 1000
        
        if result:
            self.last_results = result
            self.query_history.append({
                "query": query,
                "top_k": top_k, 
                "min_similarity": min_similarity,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results_count": result['total_results']
            })
            
            print(f"\n📊 Found {result['total_results']} results")
            if self.config['show_timing']:
                server_time = result.get('processing_time_ms', 0)
                print(f"⏱️  Server: {server_time:.1f}ms, Client: {client_time:.1f}ms")
            
            print("-" * 60)
            
            for i, doc in enumerate(result['results'], 1):
                print(f"{i}. {doc['title']}")
                print(f"   📈 Similarity: {doc['similarity_score']:.3f}")
                
                preview = doc['content_preview'][:self.config['max_preview_length']]
                print(f"   📝 Preview: {preview}...")
                
                if self.config['show_urls'] and doc.get('url'):
                    print(f"   🔗 URL: {doc['url']}")
                print()
    
    def search_history(self):
        """Display search history."""
        self.display_header("Search History")
        
        if not self.query_history:
            print("📝 No search history available")
            return
        
        print(f"📊 Total searches: {len(self.query_history)}\n")
        
        for i, search in enumerate(reversed(self.query_history[-10:]), 1):
            print(f"{i}. '{search['query']}'")
            print(f"   🕒 {search['timestamp']}")
            print(f"   📊 {search['results_count']} results (top_k={search['top_k']}, min_sim={search['min_similarity']})")
            print()
    
    def export_results(self):
        """Export last search results."""
        if not self.last_results:
            print("❌ No results to export")
            return
            
        self.display_header("Export Results")
        
        filename = f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.last_results, f, indent=2)
            print(f"✅ Results exported to: {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def batch_search(self):
        """Perform multiple searches."""
        self.display_header("Batch Search")
        
        queries = [
            "machine learning algorithms",
            "deep neural networks",
            "natural language processing", 
            "computer vision",
            "data preprocessing"
        ]
        
        print("🎯 Running sample batch queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n📍 Query {i}/{len(queries)}: '{query}'")
            self.perform_search(query, 3, 0.1)
            
            if i < len(queries):
                input("Press Enter to continue...")

    # =========================
    # 🔄 PIPELINE MANAGEMENT
    # =========================
    
    def pipeline_menu(self):
        """Pipeline management operations."""
        while True:
            self.display_menu("Pipeline Management", [
                "🔄 Reload Pipeline",
                "🔥 Force Regenerate", 
                "📊 Cache Information",
                "🧹 Clear Session Data"
            ])
            
            choice = self.get_user_choice(4)
            if choice == 0:
                break
            elif choice == 1:
                self.reload_pipeline(False)
            elif choice == 2:
                self.reload_pipeline(True)
            elif choice == 3:
                self.cache_info()
            elif choice == 4:
                self.clear_session()
    
    def reload_pipeline(self, force_regenerate: bool):
        """Reload the pipeline."""
        action = "Force Regenerate" if force_regenerate else "Reload Pipeline"
        self.display_header(action)
        
        print(f"🔄 {action} in progress...")
        print("   This may take several minutes...")
        
        endpoint = f"/reload?force_regenerate={'true' if force_regenerate else 'false'}"
        result = self.make_request(endpoint, "POST")
        
        if result:
            print(f"✅ {result.get('message', 'Operation completed')}")
            if force_regenerate:
                print("⚠️  Embeddings will be regenerated in the background")
        
    def cache_info(self):
        """Display cache information."""
        self.display_header("Cache Information")
        
        # Check if data directory exists
        data_dir = "data"
        if os.path.exists(data_dir):
            print(f"📁 Data directory: {os.path.abspath(data_dir)}")
            
            files = ["wikipedia_corpus.json", "wikipedia_embeddings.npy", "wikipedia_embeddings_metadata.json"]
            for filename in files:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    print(f"   ✅ {filename}: {size:.2f} MB (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    print(f"   ❌ {filename}: Not found")
        else:
            print(f"❌ Data directory not found: {data_dir}")
    
    def clear_session(self):
        """Clear session data."""
        self.display_header("Clear Session Data")
        
        confirm = input("🗑️  Clear search history and cached results? (y/N): ").strip().lower()
        if confirm == 'y':
            self.query_history.clear()
            self.last_results = None
            print("✅ Session data cleared")
        else:
            print("🚫 Operation cancelled")

    # =========================
    # ⚙️ CONFIGURATION
    # =========================
    
    def settings_menu(self):
        """Configuration and settings."""
        while True:
            self.display_menu("Settings & Configuration", [
                "🔧 View Current Settings",
                "📊 Modify Default Parameters",
                "🎨 Display Options",
                "🌐 Server Configuration",
                "💾 Save Configuration",
                "🔄 Reset to Defaults"
            ])
            
            choice = self.get_user_choice(6)
            if choice == 0:
                break
            elif choice == 1:
                self.view_settings()
            elif choice == 2:
                self.modify_defaults()
            elif choice == 3:
                self.display_options()
            elif choice == 4:
                self.server_config()
            elif choice == 5:
                self.save_config()
            elif choice == 6:
                self.reset_config()
    
    def view_settings(self):
        """Display current settings."""
        self.display_header("Current Settings")
        
        print(f"🌐 Server URL: {self.base_url}")
        print(f"\n📊 Default Search Parameters:")
        print(f"   Top K: {self.config['default_top_k']}")
        print(f"   Min Similarity: {self.config['default_min_similarity']}")
        print(f"\n🎨 Display Options:")
        print(f"   Show URLs: {self.config['show_urls']}")
        print(f"   Show Timing: {self.config['show_timing']}")
        print(f"   Max Preview Length: {self.config['max_preview_length']}")
    
    def modify_defaults(self):
        """Modify default search parameters."""
        self.display_header("Modify Default Parameters")
        
        try:
            top_k = input(f"📊 Default Top K (current: {self.config['default_top_k']}): ").strip()
            if top_k:
                self.config['default_top_k'] = int(top_k)
                
            min_sim = input(f"📈 Default Min Similarity (current: {self.config['default_min_similarity']}): ").strip()
            if min_sim:
                self.config['default_min_similarity'] = float(min_sim)
                
            print("✅ Default parameters updated")
            
        except ValueError:
            print("❌ Invalid values entered")
    
    def display_options(self):
        """Modify display options."""
        self.display_header("Display Options")
        
        show_urls = input(f"🔗 Show URLs (current: {self.config['show_urls']}) [y/n]: ").strip().lower()
        if show_urls in ['y', 'n']:
            self.config['show_urls'] = show_urls == 'y'
            
        show_timing = input(f"⏱️  Show Timing (current: {self.config['show_timing']}) [y/n]: ").strip().lower()
        if show_timing in ['y', 'n']:
            self.config['show_timing'] = show_timing == 'y'
            
        try:
            preview_len = input(f"📝 Max Preview Length (current: {self.config['max_preview_length']}): ").strip()
            if preview_len:
                self.config['max_preview_length'] = int(preview_len)
        except ValueError:
            print("❌ Invalid preview length")
            
        print("✅ Display options updated")
    
    def server_config(self):
        """Configure server connection."""
        self.display_header("Server Configuration")
        
        new_url = input(f"🌐 Server URL (current: {self.base_url}): ").strip()
        if new_url:
            self.base_url = new_url.rstrip('/')
            print(f"✅ Server URL updated to: {self.base_url}")
    
    def save_config(self):
        """Save configuration to file."""
        config_data = {
            'base_url': self.base_url,
            'config': self.config
        }
        
        try:
            with open('rag_cli_config.json', 'w') as f:
                json.dump(config_data, f, indent=2)
            print("✅ Configuration saved to: rag_cli_config.json")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def reset_config(self):
        """Reset to default configuration."""
        confirm = input("🔄 Reset all settings to defaults? (y/N): ").strip().lower()
        if confirm == 'y':
            self.config = {
                "default_top_k": 5,
                "default_min_similarity": 0.1,
                "show_urls": True,
                "show_timing": True,
                "max_preview_length": 150
            }
            self.base_url = "http://localhost:8000"
            print("✅ Configuration reset to defaults")
        else:
            print("🚫 Reset cancelled")

    # =========================
    # 📚 DOCUMENTATION & HELP
    # =========================
    
    def help_menu(self):
        """Help and documentation."""
        while True:
            self.display_menu("Help & Documentation", [
                "📖 API Endpoints Reference",
                "🎯 Sample Queries",
                "🌐 Open Web Documentation",
                "📋 Keyboard Shortcuts",
                "ℹ️  About This Client"
            ])
            
            choice = self.get_user_choice(5)
            if choice == 0:
                break
            elif choice == 1:
                self.api_reference()
            elif choice == 2:
                self.sample_queries()
            elif choice == 3:
                self.open_web_docs()
            elif choice == 4:
                self.keyboard_shortcuts()
            elif choice == 5:
                self.about_client()
    
    def api_reference(self):
        """Display API endpoints reference."""
        self.display_header("API Endpoints Reference")
        
        endpoints = [
            ("GET /", "🏠 Root information and available endpoints"),
            ("GET /health", "🏥 Health check and pipeline status"),
            ("GET /status", "📊 Detailed pipeline and system status"),
            ("POST /search", "🔍 Semantic search with query parameters"),
            ("POST /reload", "🔄 Reload pipeline data and embeddings")
        ]
        
        for endpoint, description in endpoints:
            print(f"🔗 {endpoint}")
            print(f"   {description}")
            print()
    
    def sample_queries(self):
        """Display sample queries by category."""
        self.display_header("Sample Queries by Category")
        
        categories = {
            "🤖 AI & Machine Learning": [
                "What is computer vision?",
                "machine learning algorithms", 
                "deep neural networks",
                "artificial intelligence applications"
            ],
            "📊 Data Science": [
                "data preprocessing techniques",
                "statistical modeling",
                "feature engineering", 
                "model evaluation metrics"
            ],
            "🧠 Advanced Topics": [
                "natural language processing",
                "reinforcement learning",
                "neural network architectures",
                "predictive analytics"
            ]
        }
        
        for category, queries in categories.items():
            print(f"\n{category}:")
            for query in queries:
                print(f"   • {query}")
    
    def open_web_docs(self):
        """Open web documentation."""
        self.display_header("Web Documentation")
        
        docs_url = f"{self.base_url}/docs"
        print(f"🌐 Interactive API Documentation:")
        print(f"   {docs_url}")
        print(f"\n📖 Alternative Documentation:")
        print(f"   {self.base_url}/redoc")
        
        try:
            import webbrowser
            open_browser = input(f"\n🌐 Open {docs_url} in browser? (y/N): ").strip().lower()
            if open_browser == 'y':
                webbrowser.open(docs_url)
                print("✅ Documentation opened in browser")
        except ImportError:
            print("🔗 Copy the URL above to open in your browser")
    
    def keyboard_shortcuts(self):
        """Display keyboard shortcuts."""
        self.display_header("Keyboard Shortcuts")
        
        shortcuts = [
            ("Ctrl+C", "🚫 Cancel current operation / Exit"),
            ("Enter", "✅ Confirm selection / Continue"),
            ("0", "🔙 Back to previous menu / Exit"),
            ("1-9", "📋 Select menu option"),
        ]
        
        for shortcut, description in shortcuts:
            print(f"⌨️  {shortcut:<10} {description}")
    
    def about_client(self):
        """Display information about the client."""
        self.display_header("About RAG API CLI Client")
        
        print("🔧 RAG API CLI Client v1.0")
        print("📅 Built: September 2025")
        print("🎯 Purpose: Professional command-line interface for RAG API")
        print("\n✨ Features:")
        features = [
            "Complete API access via intuitive menus",
            "Interactive search with customizable parameters", 
            "Pipeline management and monitoring",
            "Configuration management and persistence",
            "Search history and result export",
            "Comprehensive help and documentation"
        ]
        
        for feature in features:
            print(f"   • {feature}")
        
        print(f"\n🌐 Connected to: {self.base_url}")

    # =========================
    # 🎮 MAIN MENU & RUNNER
    # =========================
    
    def main_menu(self):
        """Display main menu and handle navigation."""
        while True:
            print(f"\n{'='*60}")
            print("🔧 RAG API CLI Client")
            print(f"{'='*60}")
            print(f"🌐 Connected to: {self.base_url}")
            
            if self.query_history:
                print(f"📊 Session: {len(self.query_history)} searches performed")
            
            self.display_menu("Main Menu", [
                "🏠 Server Status & Health",
                "🔍 Interactive Search", 
                "🔄 Pipeline Management",
                "⚙️  Settings & Configuration",
                "📚 Help & Documentation"
            ])
            
            choice = self.get_user_choice(5)
            
            if choice == 0:
                print("\n👋 Thank you for using RAG API CLI Client!")
                break
            elif choice == 1:
                self.server_status_menu()
            elif choice == 2:
                self.search_menu()
            elif choice == 3:
                self.pipeline_menu()
            elif choice == 4:
                self.settings_menu()
            elif choice == 5:
                self.help_menu()
    
    def run(self):
        """Main entry point for the CLI client."""
        print("🚀 RAG API CLI Client Starting...")
        
        # Try to load saved configuration
        try:
            if os.path.exists('rag_cli_config.json'):
                with open('rag_cli_config.json', 'r') as f:
                    saved_config = json.load(f)
                    self.base_url = saved_config.get('base_url', self.base_url)
                    self.config.update(saved_config.get('config', {}))
                print("✅ Configuration loaded from file")
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
        
        # Test initial connection
        print(f"🔌 Testing connection to {self.base_url}...")
        root_info = self.make_request("/")
        
        if root_info:
            print(f"✅ Connected to: {root_info.get('message', 'RAG API Server')}")
            self.main_menu()
        else:
            print("❌ Could not connect to server")
            print("💡 Make sure the server is running with: python src/main.py")
            
            # Ask if user wants to continue anyway
            continue_anyway = input("\n🤔 Continue anyway? (y/N): ").strip().lower()
            if continue_anyway == 'y':
                self.main_menu()


def main():
    """Entry point for the CLI application."""
    try:
        client = RAGAPIClient()
        client.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please report this issue if it persists.")


if __name__ == "__main__":
    main()
