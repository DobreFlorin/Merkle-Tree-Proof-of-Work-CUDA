# Merkle-Tree-Proof-of-Work-CUDA

Implementarea este urmatoarea:
Pentru contrsuiea arborelui Merkle:
Functia  contrsuieste arborele Merkle dintr o lista de tranzactii, afland Merkle root-ul:
in uramtoarele etape:
-alocare memorie pentru tranzactii , hash-uri, rezultate partiale
-copierea datelor de tranzactii de la host catre device
-calcularea hash-urilor
-construirea propriu-zisa a arborelui:se combina perechile folosind combine_hashes_kernel pana 
obtinem un singur hash(merkle-root-ul);daca numarul e impar ultimul hash va fi duplicat.
-copiere rezultat 
-dezalocare memorie

Pentru gasire nonce:
-alocare memorie pentru dificultate, block_content, block_hash,  found_nonce, found_flag
-se copiaza  difficulty si block_content in device. found_flag  = 0
-configuram 256 de threaduri per block si 65,535 blocks si aloc nonce urile per thread
-functia de calculare nonce-uri in paralel
-sincronizarea threadurilor
-verificarea rezultatului: se copiaza found_flag la host; daca este 1 returneaza 0, altfel 1
-eliberare memorie alocata
