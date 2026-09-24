# Update and rebuild the GPU-enabled Docker containers for ssTranscriber
git pull && docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec -u root sstranscriber sh -c '/sbin/ldconfig /usr/lib/wsl/drivers/*'

# Update and rebuild the non-GPU Docker containers for ssTranscriber
# git pull && docker compose up --build -d