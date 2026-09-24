git pull && docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec -u root sstranscriber sh -c '/sbin/ldconfig /usr/lib/wsl/drivers/*'