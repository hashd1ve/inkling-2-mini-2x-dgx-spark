#!/bin/bash
# Inkling 2 Mini repartido entre las dos Sparks (TP=2 por el cable ConnectX, NCCL sobre RoCE). Se lanza desde el Mac.
# Cabeza (rango 0, API en :PORT) en HEAD; el otro nodo va sin API (--headless).
# Uso: HEAD=<head> HEAD_IP=<ip> WORKER=<worker> WORKER_IP=<ip> SPEC=7 ADAPT=1 RDMA=1 GEMV12=1 bash serve/serve_tp2.sh      SPEC=0 desactiva DSpark.   STOP=1 bash serve/serve_tp2.sh para pararlo.
SPEC=${SPEC:-7}; PORT=${PORT:-8010}; NAME=${NAME:-inkling-tp2}; MEM=${MEM:-0.75}; MAXLEN=${MAXLEN:-524288}
# HEAD / WORKER: nombres ssh de los dos nodos. HEAD_IP / WORKER_IP: su IPv4 en el enlace ConnectX. Obligatorios.
HEAD=${HEAD:?falta HEAD (nombre ssh del nodo con la API)}; WORKER=${WORKER:?falta WORKER (nombre ssh del otro nodo)}
if [ "${STOP:-0}" != "1" ]; then
  HEAD_IP=${HEAD_IP:?falta HEAD_IP (IPv4 del enlace en HEAD)}; WORKER_IP=${WORKER_IP:?falta WORKER_IP (IPv4 del enlace en WORKER)}
fi
IMG=vllm/vllm-openai:inkling-2-mini
# ADAPT=1: verificación adaptativa de DSpark (monta patch/adaptive, copiado en ~/inkling-tp2/adaptive de cada Spark).
if [ "${ADAPT:-0}" = "1" ]; then
  P=/usr/local/lib/python3.14/dist-packages/vllm/models; A='$HOME/inkling-tp2/adaptive'
  SPEC_EXTRA="${SPEC_EXTRA:-},\"enable_adaptive_verification\":true"
  # Coste fijo por paso y por contexto que el perfil de vLLM no ve (sin ellos recorta de más; ver README).
  EXTRA_ENV="${EXTRA_ENV:-} -e ADAPT_FIXED_MS=${ADAPT_FIXED_MS:-15} -e ADAPT_CTX_MS_PER_1K=${ADAPT_CTX_MS_PER_1K:-0.09} -e DRAFT_SCALE=${DRAFT_SCALE:-1}"
  EXTRA_DOCKER="${EXTRA_DOCKER:-} -v $A/speculator.py:$P/inkling2/speculator.py:ro -v $A/sconv_swa_attn.py:$P/inkling/nvidia/sconv_swa_attn.py:ro -v $A/attention.py:$P/inkling2/nvidia/attention.py:ro -v $A/adaptive_verification.py:/usr/local/lib/python3.14/dist-packages/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:ro"
fi
# RDMA=1: all-reduce propio por RDMA para bf16 pequeños en vez de NCCL (patch/rdma, copiado en ~/inkling-tp2/rdma).
if [ "${RDMA:-0}" = "1" ]; then
  D=/usr/local/lib/python3.14/dist-packages/vllm/distributed/device_communicators; R='$HOME/inkling-tp2/rdma'
  EXTRA_ENV="${EXTRA_ENV:-} -e RDMA_AR=1"
  EXTRA_DOCKER="${EXTRA_DOCKER:-} -v $R/cuda_communicator.py:$D/cuda_communicator.py:ro -v $R/rdma_all_reduce.py:$D/rdma_all_reduce.py:ro -v $R/rdma_xchg.c:$D/rdma_xchg.c:ro -v $R/prof_hook.py:$D/prof_hook.py:ro"
fi
# GEMV12=1: pesos bf16 leídos en 12 bits sin pérdida, salida idéntica bit a bit (patch/gemv12, en ~/inkling-tp2/gemv12).
if [ "${GEMV12:-0}" = "1" ]; then
  EXTRA_ENV="${EXTRA_ENV:-} -e GEMV12=1"
  EXTRA_DOCKER="${EXTRA_DOCKER:-} -v \$HOME/inkling-tp2/gemv12/gemv.py:/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia/ops/gemv.py:ro"
fi

if [ "${STOP:-0}" = "1" ]; then
  for h in $HEAD $WORKER; do ssh $h "docker rm -f $NAME >/dev/null 2>&1"; done; echo parado; exit 0
fi

ARGS=(thinkingmachines/Inkling-2-Mini --tensor-parallel-size 2 --nnodes 2 --master-addr $HEAD_IP --master-port 29501
      --max-model-len $MAXLEN --gpu-memory-utilization $MEM
      --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling)
[ "$SPEC" != "0" ] && ARGS+=(--speculative-config "'{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC,\"draft_sample_method\":\"probabilistic\"${SPEC_EXTRA:-}}'")

# GID 5 = RoCE v2 con la IPv4 del enlace en los dos puertos (rocep1s0f1, roceP2p1s0f1).
# GPUDirect RDMA no funciona en el GB10 (NCCL lo registra como GDR 0): ~50 µs por all-reduce pequeño.
NCCL="-e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e GLOO_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
      -e NCCL_IB_GID_INDEX=5 -e NCCL_CUMEM_ENABLE=0 -e HF_HUB_OFFLINE=1 ${EXTRA_ENV:-}"
C='$HOME/.cache/inkling-lab'
DOCKER="docker run -d --name $NAME --gpus all --ipc=host --network host --privileged --ulimit memlock=-1:-1
        --device /dev/infiniband:/dev/infiniband $NCCL ${EXTRA_DOCKER:-}
        -v \$HOME/.cache/huggingface:/root/.cache/huggingface -v $C/vllm:/root/.cache/vllm -v $C/triton:/root/.triton
        -v $C/flashinfer:/root/.cache/flashinfer -v $C/torch:/root/.cache/torch"
DOCKER=$(echo $DOCKER)  # en una sola línea: ssh se lo pasa tal cual al shell remoto

for h in $HEAD $WORKER; do ssh $h "docker rm -f $NAME >/dev/null 2>&1; mkdir -p $C/vllm $C/triton $C/flashinfer $C/torch"; done
ssh $WORKER "$DOCKER -e VLLM_HOST_IP=$WORKER_IP $IMG ${ARGS[*]} --node-rank 1 --headless $EXTRA" >/dev/null
ssh $HEAD "$DOCKER -e VLLM_HOST_IP=$HEAD_IP $IMG ${ARGS[*]} --node-rank 0 --port $PORT $EXTRA" >/dev/null

end=$(( $(date +%s) + 1800 ))
until ssh $HEAD "curl -sf localhost:$PORT/health" >/dev/null; do
  [ $(date +%s) -gt $end ] && { echo TOPE; exit 1; }
  for h in $HEAD $WORKER; do
    ssh $h "docker ps -q -f name=$NAME" | grep -q . || { echo "CAIDO en $h"; ssh $h "docker logs --tail 40 $NAME"; exit 1; }
  done
  sleep 15
done
echo "listo en $(( $(date +%s) - end + 1800 )) s"
