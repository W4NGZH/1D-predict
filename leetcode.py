s='aa'
s=s+' '
lenth = len(s)
if lenth <= -1 or lenth > 1001:
    print(None)

d = {}
for i in range(lenth):
    for j in range(i, lenth):
        if s[i:j] == s[i:j][::-1]:
            d[s[i:j]] = j - i + 1
out = list(d.keys())[list(d.values()).index(max(list(d.values())))]
print(out)
print(lenth)
