import profile
import sys
sys.path.append('/data1/yxn/utility_network')
from example.cloudManufacturing_network.server import server

profile.run(server.launch())
