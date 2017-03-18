from .setup import * 


def load_processed_patches(tile_number, tile_level_index):    
    raw_X = []
    tissue_labels = []
    for t in tissues_types:
        print (t)
        [rX, tl] = pickle.load(open('../data/processed/data_{}_{}_{}.py'.format(t,tile_number,tile_level_index), 'rb'))
        raw_X += rX
        tissue_labels += tl

    X = (np.array(raw_X,dtype=np.float16) / 255)
    return raw_X, X, tissue_labels

def plot_histories():
    histories1 = pickle.load(open('models/histories_1_-4.py','rb'))
    histories5 = pickle.load(open('models/histories_5_-4.py','rb'))
    histories10 = pickle.load(open('models/histories_10_-4.py','rb'))
    red_patch = mpatches.Patch(color='red')
    blue_patch = mpatches.Patch(color='blue')
    green_patch = mpatches.Patch(color='green')
    simArtist = plt.Line2D((0,1),(0,0), color='k', linestyle='dotted')
    anyArtist = plt.Line2D((0,1),(0,0), color='k')

    f,a = plt.subplots(1,2, figsize=(15,5))
    f.suptitle("Patch size 1024x1024",size=20)
    a[0].set_title("Final layer")
    a[0].set_xlabel('Epoch')
    a[0].set_ylabel('Accuracy')
    a[0].plot(histories1[0]['val_acc'],c='red')
    a[0].plot(histories1[0]['acc'], c='red',ls='dotted')
    a[0].plot(histories5[0]['val_acc'],c='blue')
    a[0].plot(histories5[0]['acc'], c='blue',ls='dotted')
    a[0].plot(histories10[0]['val_acc'],c='green')
    a[0].plot(histories10[0]['acc'], c='green',ls='dotted')

    handles = [red_patch,blue_patch,green_patch, simArtist,anyArtist]
    labels = ['100 per class', '500 per class', '1000 per class','Train','Test']
    a[0].legend(handles=handles,labels=labels, loc='upper left')

    a[1].set_title("Fine tuning")
    a[1].set_xlabel('Epoch')
    a[1].set_ylabel('Accuracy')
    a[1].plot(histories1[1]['val_acc'],c='red')
    a[1].plot(histories1[1]['acc'],c='red',ls='dotted')
    a[1].plot(histories5[1]['val_acc'],c='blue')
    a[1].plot(histories5[1]['acc'],c='blue',ls='dotted')
    a[1].plot(histories10[1]['val_acc'],c='green')
    a[1].plot(histories10[1]['acc'],c='green',ls='dotted')
    a[1].legend(handles=handles,labels=labels, loc='upper left')

    # a[1].legend(handles=[red_patch, simArtist,anyArtist],labels=[1,2,3])
    # handles, labels = a[1].get_legend_handles_labels()


def build_empty_model():
    inception_model = InceptionV3(weights='imagenet', include_top=False)

    x = inception_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inception_model.input, output=predictions)
    return model

def visualise_lungs():
    f,a = plt.subplots(5,8, figsize=(15,10))
    for (i,ax) in enumerate(a.flatten()):
        ax.imshow(np.array(raw_X[i], dtype=np.float32))
        ax.set_title(tissue_labels[i])
        ax.axis('off')

def visualise_random_patches():
    random_idx = np.random.choice(len(tissue_labels),40)
    f,a = plt.subplots(5,8, figsize=(15,10))
    for (i,ax) in enumerate(a.flatten()):
        image = np.array(raw_X[random_idx[i]], dtype=np.float32)
        max_intensity = np.max(image.flatten())
        ax.imshow(1- (image / max_intensity))
        ax.set_title(tissue_labels[random_idx[i]])
        ax.axis('off')
        
def get_components_tsne(tsne, full_representations):
    try:
        components_tsne = pickle.load(open('intermediate/components_tsne.py','rb'))
        return components_tsne
        
    except FileNotFoundError:
        components_tsne = tsne.fit_transform(full_representations)
        pickle.dump(components_tsne, open('intermediate/components_tsne.py','wb'))
        return components_tsne
    
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])
    return activations


def plot_activations(activations):
    nb_filters = activations.shape[-1]
    print (nb_filters)
    random_idx = np.random.choice(range(nb_filters), 16)
    f,a = plt.subplots(4,4,figsize=(6,6))
    for i in range(16):
        a.flatten()[i].imshow(activations[:,:,random_idx[i]])
        a.flatten()[i].axis('off')
    
def visualise_3d_tsne(full_representations):
    red_patch = mpatches.Patch(color='red')
    black_patch = mpatches.Patch(color='black')
    purple_patch = mpatches.Patch(color='purple')
    green_patch = mpatches.Patch(color='green')
    blue_patch = mpatches.Patch(color='blue')
    yellow_patch = mpatches.Patch(color='yellow')
    orange_patch = mpatches.Patch(color='orange')
    pink_patch = mpatches.Patch(color='pink')
    cyan_patch = mpatches.Patch(color='cyan')
    grey_patch = mpatches.Patch(color='grey')
    c = ['red','black','purple','green','blue','yellow','orange','pink','cyan','grey']
    labels = ['100 per class', '500 per class', '1000 per class','2000 per class', '3000 per class', '4000 per class','5000 per class','Train','Test']
    handles = [red_patch,black_patch,purple_patch,green_patch,blue_patch,yellow_patch, orange_patch, pink_patch, cyan_patch, grey_patch]
    pca = PCA(n_components=3)
    components_pca = pca.fit_transform(full_representations)
    components_tsne = get_components_tsne(TSNE(n_components=3, perplexity=30, n_iter=5000), full_representations)
    f = plt.figure(figsize=(15,7))
    ax = f.add_subplot(1, 2, 1, projection='3d')
    ax.scatter([x[0] for x in components_pca], [x[1] for x in components_pca], [x[2] for x in components_pca], c=[c[np.argmax(x)] for x in y], alpha=0.5, s=3)
    ax.legend(handles=handles,labels=list(classes), loc='lower right')
    ax = f.add_subplot(1, 2, 2, projection='3d')
    ax.scatter([x[0] for x in components_tsne], [x[1] for x in components_tsne], [x[2] for x in components_tsne], c=[c[np.argmax(x)] for x in y], alpha=0.5, s=3)
    ax.legend(handles=handles,labels=list(classes), loc='lower right')

def get_tissue_components_tsne(tsne, tissue_representations, tissue):
    try:
        components_tsne = pickle.load(open('intermediate/{}_components_tsne.py'.format(tissue),'rb'))
        return components_tsne
        
    except FileNotFoundError:
        components_tsne = tsne.fit_transform(tissue_representations)
        pickle.dump(components_tsne, open('intermediate/{}_components_tsne.py'.format(tissue),'wb'))
        return components_tsne

def visualise_tissue_tsne(full_representations, tissue = 'Brain - Cerebellum'):
    c = (['orange']*10 + ['blue']*10 + ['red']*10 + ['grey']*10 + ['cyan']*10 + ['purple']*10 + ['green']*10 + ['black']*10 + ['orange']*10 + ['yellow']*10)
    # tissue = 'Breast - Mammary Tissue'

    tissue_idx = np.array([classes[np.argmax(x)] for x in y]) == tissue

    tissue_representations = full_representations[tissue_idx]
    print (len(tissue_representations))
    randint = np.random.randint(90)
    tissue_representations = tissue_representations[randint:randint+100]

    tsne = TSNE(n_components=2, perplexity=100,n_iter=5000)
    tissue_components_tsne = components_tsne = tsne.fit_transform(tissue_representations)
    pca = PCA(n_components=2)
    tissue_components_pca = pca.fit_transform(tissue_representations)

    f = plt.figure(figsize=(17,7))
    ax = f.add_subplot(1, 2, 1)
    ax.scatter([x[0] for x in tissue_components_pca], [x[1] for x in tissue_components_pca],c=c, alpha=1, s=20)
    # ax.legend(handles=handles,labels=list(classes), loc='lower right')
    ax = f.add_subplot(1, 2, 2)
    ax.scatter([x[0] for x in tissue_components_tsne], [x[1] for x in tissue_components_tsne],c=c, alpha=1, s=20)
    # ax.legend(handles=handles,labels=list(classes), loc='lower right')


def plot_confusion_matrix(y_test, y_pred):
    cm = ConfusionMatrix([np.argmax(x) for x in y_test], [np.argmax(x) for x in y_pred])

    cm.plot()
    plt.xticks(range(10),tissues_types)
    plt.yticks(range(10),tissues_types)

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]



def download_tissues(random_tissue_ID_lookup):
    for t4 in list(random_tissue_ID_lookup.keys()):
        random_image_IDs = random_tissue_ID_lookup[t4]
        try:
            os.mkdir(os.path.join('data',t4))
        except:
            print('Directory exists')
        print('Downloading tissue: {}'.format(t4))
        download(random_image_IDs, os.path.join('data',t4))

def sample_tiles_from_image(tile_size,tile_number,image_path):
    #Sample n tiles of size mxm from image
    sampled_tiles = []
    try:
        slide = open_slide(os.path.join(GTEx_directory,image_path))
        tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
        tile_level = range(len(tiles.level_tiles))[tile_level_index]
        tile_dims = tiles.level_tiles[tile_level_index]
    #     f,a = plt.subplots(4,4,figsize=(10,10))
        count = 0
        
        t = time.time()
        # expect sampling rate to be at least 1 tile p/s. If time take is greater than this, move to next image.
    #         
        while (count < tile_number and (time.time() - t < tile_number * 2)):
    #             print (time.time() - t)
            #retreive tile

            tile = tiles.get_tile(tile_level, (np.random.randint(tile_dims[0]), np.random.randint(tile_dims[1])))
            image = 255 - np.array(tile.getdata(), dtype=np.float32).reshape(tile.size[0],tile.size[1],3)
            #calculate mean pixel intensity
            mean_pixel = np.mean(image.flatten())
            image = imresize(image,(299,299))
            if mean_pixel < 20:
                continue
            elif mean_pixel >= 20:
    #             a.flatten()[count].axis('off')
    #             a.flatten()[count].set_title(mean_pixel, size=5)
    #             a.flatten()[count].imshow(image)
                sampled_tiles.append(image)
                count += 1
            
        if (time.time() - t > tile_number * 2):
            print("Timeout")
    except Exception as e:
        # print (sys.exc_info())
        print("Error")
        
    return sampled_tiles
