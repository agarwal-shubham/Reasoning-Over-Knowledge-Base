# Reasoning Over Knowledge Base
#
#
#
>  The project reimplements the architecture of the paper Reasoning with Neural Tensor Networks for Knowledge Base Completion in Torch framework, achieving similar accuracy results with an elegant implementation in a modern language. 

>The project's goal is of predicting the likely truth of additional​ facts based on existing facts in a KB(knowledge base) and giving the system reasoning capability. For instance, when told that a new species of monkeys has been discovered, a person does not need to find textual​ evidence to know that this new monkey, too, will have legs.
#  
#     
The goal of our approach is to be able to state whether two entities (e1 , e2 ) are in a certain relationship R. For instance, whether the relationship (e1 , R, e2 ) = (Bengal tiger, has part, tail ) is true and with what certainty. To this end, we define a set of parameters indexed by R for each relation’s scoring function. Let e1 , e2 ∈ R be the vector representations (or features) of the two entities. For now we can assume that each value of this vector is randomly initialized to a small uniformly random number.  
The NTN replaces a standard linear neural network layer with a bilinear tensor layer that directly relates the two entity vectors across multiple dimensions.

- All models are trained with contrastive max-margin objective functions. The main idea is that each triplet in the training set Ti = (ei1, Ri, ei2 ) should receive a higher score than a triplet in which one of the entities is replaced with a random entity.
- Each relation has its associated neural tensor net parameters. We call the corrupted triplet as Tci = (ei1 , Ri , ec ), where we sampled entity ec randomly from the set of all entities that can appear at that position in that relation.
- We use mini-batched L-BFGS for optimization which converges to a local optimum of our non-convex objective functionposition in that relation.
- Experiments are conducted on both WordNet and FreeBase to predict whether some relations hold using other facts in the database. Our goal is to predict correct facts in the form of relations (e1 , R, e2 ) in the testing data. This could be seen as answering questions such as Does a dog have a tail?, using the scores g(dog, has part, tail) computed by the various models.
- We use the development set to find a threshold TR for each relation such that if g(e1 , R, e2 ) ≥ TR , the relation (e1 , R, e2 ) holds, otherwise it does not hold.
The final accuracy is based on how many triplets are classified correctly. 
For results and stepwise processing follow the links below:-

- [![Presentation link](http://www.slideshare.net/ShubhamAgarwal211/reasoning-over-knowledge-base-60968302/ShubhamAgarwal211/reasoning-over-knowledge-base-60968302)]()
- [![WebPage link](http://darsh510.github.io/IREPROJ/)]()
- [![YouTube video link](https://www.youtube.com/watch?edit=vd&v=_fCuyWF4vA8)](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)
