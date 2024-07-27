import argparse
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)  # Parent directory where is the fedstellar module
import fedstellar
from fedstellar.controller import Controller

argparser = argparse.ArgumentParser(
    description="Controller of Fedstellar platform", add_help=False
)

argparser.add_argument(
    "--grafanaport",
    dest="grafanaport",
    default=6040,
    help="Grafana port (default: 6040)",
) 

argparser.add_argument(
    "--lokiport",
    dest="lokiport",
    default=6010,
    help="Loki port (default: 6010)",
) 

argparser.add_argument(
     "--waf",
     dest="waf",
     action="store_true",
     default=False,
     help="Run WAF",
 )

argparser.add_argument(
    "--wafport",
    dest="wafport",
    default=6050,
    help="WAF port (default: 6050)",
)   

argparser.add_argument(
    "-wp",
    "--webport",
    dest="webport",
    default=6060,
    help="Frontend port (default: 6060)",
)
argparser.add_argument(
    "-st",
    "--stop",
    dest="stop",
    nargs="?",
    const='all', # If no argument is given, stop all
    default=None,
    help="Stop Fedstellar platform or nodes only (use '--stop nodes' to stop only the nodes)",
)
argparser.add_argument(
    "-sp",
    "--statsport",
    dest="statsport",
    default=6065,
    help="Statistics port (default: 6065)",
)
argparser.add_argument(
    "-s", "--simulation", action="store_false", dest="simulation", help="Run simulation"
)
argparser.add_argument(
    "-c",
    "--config",
    dest="config",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config"),
    help="Config directory path",
)
argparser.add_argument(
    "-l",
    "--logs",
    dest="logs",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
    help="Logs directory path",
)
argparser.add_argument(
    "-m",
    "--models",
    dest="models",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    help="Models directory path",
)
# Path to the file in same directory as this file
argparser.add_argument(
    "-e",
    "--env",
    dest="env",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    help=".env file path",
)
argparser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    default=False,
    help="Debug mode",
)
argparser.add_argument(
    "-v",
    "--version",
    action="version",
    version="%(prog)s " + fedstellar.__version__,
    help="Show version",
)
argparser.add_argument(
    "-a",
    "--about",
    action="version",
    version="Created by Enrique Tomás Martínez Beltrán",
    help="Show author",
)
argparser.add_argument(
    "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show help"
)

args = argparser.parse_args()

"""
Code for deploying the controller 
"""
if __name__ == "__main__":
    
    if args.stop == 'all':
        Controller.stop()
    elif args.stop == 'nodes':
        Controller.stop_nodes()
    
    Controller(args).start()
