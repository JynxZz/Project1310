#!/bin/bash
echo "hello welcome to your pinging assistant"
echo -e "\npinging all"
declare -a APIS
APIS=("https://sdg-classifier.streamlit.app")
for val in "${APIS[@]}"; do
if ! curl -o /dev/null -s -w "%{http_code}\n" "$val"; then
echo "$val as failed !!"
fi
done
