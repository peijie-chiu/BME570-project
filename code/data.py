from utils import generate_object
from imaging import generate_recon

## generate object
num_objects = 500
generate_object(num_objects)

## generate reconstructed image
generate_recon(num_objects)