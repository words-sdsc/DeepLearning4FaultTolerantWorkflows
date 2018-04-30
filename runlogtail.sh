echo "Hello: " $1
tail -n 1000000 "$1" >  "$1.short"
