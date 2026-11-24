// Intercambio RDMA entre dos nodos para el all-reduce de TP=2 (ver rdma_all_reduce.py).
// El kernel de GPU deja su fila en una ranura de envío y escribe el nº de secuencia en la cabecera; este hilo lo ve
// (espera activa), y manda con dos RDMA WRITE encadenados la carga y después la cabecera a la misma ranura del otro
// nodo, donde el kernel de GPU espera a ver la secuencia. RC garantiza que la carga llega antes que la cabecera.
#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HDR 64
#define SIGNAL_EVERY 32

static struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    uint8_t *send, *recv;
    size_t slot;
    int nslots, gid_idx, cpu;
    uint64_t raddr;
    uint32_t rkey;
    pthread_t th;
    volatile int stop;
    volatile uint64_t sent;
} g;

static void die(const char *m) { perror(m); abort(); }

// buf: 2 * nslots * slot bytes (mitad de envío, mitad de recepción). info[6]: qpn, psn, gid (2), rkey, dirección.
int xc_init(const char *dev, int gid_idx, void *buf, size_t slot, int nslots, uint64_t *info) {
    int n;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) return -1;
    for (int i = 0; i < n; i++)
        if (!strcmp(ibv_get_device_name(list[i]), dev)) g.ctx = ibv_open_device(list[i]);
    ibv_free_device_list(list);
    if (!g.ctx) return -2;
    g.pd = ibv_alloc_pd(g.ctx);
    g.cq = ibv_create_cq(g.ctx, 64, NULL, NULL, 0);
    if (!g.pd || !g.cq) return -3;
    struct ibv_qp_init_attr qi = {.send_cq = g.cq, .recv_cq = g.cq, .qp_type = IBV_QPT_RC, .sq_sig_all = 0,
                                  .cap = {.max_send_wr = 512, .max_recv_wr = 4, .max_send_sge = 1, .max_recv_sge = 1,
                                          .max_inline_data = 16}};
    g.qp = ibv_create_qp(g.pd, &qi);
    if (!g.qp) return -4;
    g.slot = slot, g.nslots = nslots, g.gid_idx = gid_idx;
    g.send = buf, g.recv = (uint8_t *)buf + slot * nslots;
    memset(buf, 0, 2 * slot * nslots);
    g.mr = ibv_reg_mr(g.pd, buf, 2 * slot * nslots, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!g.mr) return -5;
    struct ibv_qp_attr a = {.qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = 1,
                            .qp_access_flags = IBV_ACCESS_REMOTE_WRITE};
    if (ibv_modify_qp(g.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) return -6;
    union ibv_gid gid;
    if (ibv_query_gid(g.ctx, 1, gid_idx, &gid)) return -7;
    info[0] = g.qp->qp_num, info[1] = 0x1234, info[4] = g.mr->rkey, info[5] = (uint64_t)g.recv;
    memcpy(&info[2], &gid, 16);
    return 0;
}

static void *loop(void *arg) {
    if (g.cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(g.cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    uint64_t next = 1, idle = 0;
    int outstanding = 0;
    while (!g.stop) {
        size_t off = (next % g.nslots) * g.slot;
        volatile uint64_t *h = (volatile uint64_t *)(g.send + off);
        if (__atomic_load_n(h, __ATOMIC_ACQUIRE) != next) {
            if (++idle > 2000000) usleep(100);  // ~ms sin tráfico: deja de quemar el núcleo
            else __asm__ volatile("yield");
            continue;
        }
        idle = 0;
        uint64_t hdr[2] = {next, h[1]};
        struct ibv_sge s1 = {.addr = (uint64_t)h + HDR, .length = (uint32_t)hdr[1], .lkey = g.mr->lkey};
        struct ibv_sge s2 = {.addr = (uint64_t)hdr, .length = 16, .lkey = 0};
        struct ibv_send_wr w2 = {.sg_list = &s2, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE,
                                 .send_flags = IBV_SEND_INLINE, .wr = {.rdma = {g.raddr + off, g.rkey}}};
        struct ibv_send_wr w1 = {.next = &w2, .sg_list = &s1, .num_sge = 1, .opcode = IBV_WR_RDMA_WRITE,
                                 .wr = {.rdma = {g.raddr + off + HDR, g.rkey}}};
        if (next % SIGNAL_EVERY == 0) w2.send_flags |= IBV_SEND_SIGNALED, outstanding++;
        struct ibv_send_wr *bad;
        if (ibv_post_send(g.qp, &w1, &bad)) die("ibv_post_send");
        g.sent = next++;
        while (outstanding) {
            struct ibv_wc wc;
            int k = ibv_poll_cq(g.cq, 1, &wc);
            if (k < 0) die("ibv_poll_cq");
            if (k && wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "rdma_xchg: %s\n", ibv_wc_status_str(wc.status));
                abort();
            }
            outstanding -= k;
            if (outstanding < 4) break;  // como mucho 4 * SIGNAL_EVERY * 2 WR sin confirmar
        }
    }
    return NULL;
}

int xc_connect(const uint64_t *r, int cpu) {
    struct ibv_qp_attr a = {.qp_state = IBV_QPS_RTR, .path_mtu = IBV_MTU_4096, .dest_qp_num = (uint32_t)r[0],
                            .rq_psn = (uint32_t)r[1], .max_dest_rd_atomic = 1, .min_rnr_timer = 12,
                            .ah_attr = {.is_global = 1, .port_num = 1,
                                        .grh = {.sgid_index = g.gid_idx, .hop_limit = 64}}};
    memcpy(&a.ah_attr.grh.dgid, &r[2], 16);
    if (ibv_modify_qp(g.qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        return -1;
    struct ibv_qp_attr b = {.qp_state = IBV_QPS_RTS, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = 0x1234,
                            .max_rd_atomic = 1};
    if (ibv_modify_qp(g.qp, &b, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                    IBV_QP_MAX_QP_RD_ATOMIC))
        return -2;
    g.rkey = (uint32_t)r[4], g.raddr = r[5], g.cpu = cpu;
    return pthread_create(&g.th, NULL, loop, NULL);
}

uint64_t xc_sent(void) { return g.sent; }
