echo "Hello: " $1
tail -n 10000 "$1" >  "$1.short"
