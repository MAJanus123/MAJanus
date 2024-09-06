#!/bin/bash

VERSION=0.1
device=eth0
bandwidth=8Mbps
latency=40ms
packetloss=
command=slow
retcode=0

if ! which tc > /dev/null; then
  echo "Aborting: No 'tc' iptables firewall traffic conditioner found" 1>&2
  echo "This requires the Linux iptables firewall." 1>&2
  exit 1
fi

if [ $EUID -ne 0 ]; then
  echo "Aborting: you must run this as root."
  exit 2
fi


#for i in "$@"
#do
#  echo Argument: $i
#done

command=$1
bandwidth=$2
burst=$3
latency=$4

echo "command=$command"
echo "bandwidth=$bandwidth"
echo "burst=$burst"
echo "latency=$latency"

case $command in
  set)
    # Credit to Superuser for the basic commands that run this
    # Question: http://superuser.com/questions/147156/simulating-a-low-bandwidth-high-latency-network-connection-on-linux
    # Author: Justin L. http://superuser.com/users/38740/justin-l
    # Answer: http://superuser.com/a/147434/159810
    if  tc qdisc show dev $device | grep "qdisc htb 1: root" >/dev/null; then
      verb=change
      echo "Changing existing queuing discipline"
    else
      verb=add
      echo "Adding new queuing discipline"
      tc qdisc $verb dev $device root handle 1:0 htb default 1
    fi
    tc class $verb dev $device parent 1:0 classid 1:1 htb rate $bandwidth ceil $bandwidth  &&
    #tc class $verb dev $device parent 1:1 classid 1:12 htb rate $bandwidth burst $burst  &&
    tc qdisc $verb dev $device parent 1:1 netem delay $latency
    retcode=$?
    ;;
  clear)
    echo resetting queueing discipline
    tc qdisc del dev $device root
    retcode=$?
    ;;
  status)
    tc qdisc
    retcode=$?
    ;;
esac

exit $retcode