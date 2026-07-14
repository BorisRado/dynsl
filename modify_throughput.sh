#!/bin/bash
# netshape.sh — shape the container's own eth0, bidirectionally.
# Runs INSIDE the pod. Cannot affect the host or SSH.
# Usage:
#   ./netshape.sh apply <down_mbit> <up_mbit> [auto_clear_secs]
#   ./netshape.sh clear
#   ./netshape.sh status

set -u
IFACE=eth0
IFB=ifb_dl              # container-local; distinct from host ifb0/ifb1 (separate netns)

preflight() {
  command -v tc >/dev/null || { echo "ERROR: tc not found (add iproute2 to image)"; exit 1; }
  # NET_ADMIN check: a harmless op that needs the cap
  if ! tc qdisc show dev "$IFACE" >/dev/null 2>&1; then
    echo "ERROR: cannot run tc on $IFACE — is NET_ADMIN added to the pod?"; exit 1
  fi
  echo "peer (host-side) ifindex = $(cat /sys/class/net/$IFACE/iflink 2>/dev/null)"
}

clear_all() {
  tc qdisc del dev "$IFACE" root      2>/dev/null
  tc qdisc del dev "$IFACE" ingress   2>/dev/null
  tc qdisc del dev "$IFB"   root      2>/dev/null
  ip link del "$IFB"                  2>/dev/null
  echo "cleared."
}

apply() {
  local DOWN=$1 UP=$2 TTL=${3:-0}
  preflight
  clear_all   # idempotent: always start clean

  # UPLINK (pod -> server): natural egress on eth0
  tc qdisc add dev "$IFACE" root tbf rate "${UP}mbit" burst 32kbit latency 400ms

  # DOWNLINK (server -> pod): eth0 ingress -> ifb -> shape
  ip link add "$IFB" type ifb 2>/dev/null   # module is loaded host-wide; device is netns-local
  ip link set "$IFB" up
  tc qdisc add dev "$IFACE" handle ffff: ingress
  tc filter add dev "$IFACE" parent ffff: matchall action mirred egress redirect dev "$IFB"
  tc qdisc add dev "$IFB" root tbf rate "${DOWN}mbit" burst 32kbit latency 400ms

  echo "applied: down=${DOWN}Mbit up=${UP}Mbit on $IFACE"

  # Dead-man's switch (belt-and-suspenders; kubectl exec already can't be locked out)
  if [ "$TTL" -gt 0 ]; then
    ( sleep "$TTL"; \
      tc qdisc del dev "$IFACE" root 2>/dev/null; \
      tc qdisc del dev "$IFACE" ingress 2>/dev/null; \
      tc qdisc del dev "$IFB" root 2>/dev/null; \
      ip link del "$IFB" 2>/dev/null ) &
    echo "auto-clear armed: ${TTL}s"
  fi
}

case "${1:-}" in
  apply)  shift; apply "$@";;
  clear)  clear_all;;
  status) tc -s qdisc show dev "$IFACE"; tc -s qdisc show dev "$IFB" 2>/dev/null;;
  *) echo "usage: $0 {apply <down> <up> [ttl_secs] | clear | status}"; exit 1;;
esac
