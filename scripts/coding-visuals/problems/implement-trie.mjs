import { defineVisual, frame, grid, visual } from '../primitives.mjs';
const state=(active,extra={})=>grid([
 ['prefix','child edge','terminal'],['root','a -> a','no'],['a','p -> ap','no'],['ap','p -> app','no'],['app','l -> appl','yes'],['appl','e -> apple','no'],['apple','none','yes'],
],[{row:['root','a','ap','app','appl','apple'].indexOf(active)+1,col:0,label:'node',tone:'focus',key:'trie-cursor'}],{...extra});
const draft=visual('Each root-to-node path is one shared prefix; terminal state distinguishes a complete word from a mere prefix.',[
 frame('Start with an empty root','The root has no child before insert(\"app\").',state('root',{operation:'insert(\"app\")',existing:'root only'}),'initialize-root'),
 frame('Create prefix a','setdefault creates the root child keyed by a.',state('a',{path:'root -a-> a'}),'insert-a'),
 frame('Create prefix ap','From a, setdefault creates child p for prefix ap.',state('ap',{path:'root -a-> a -p-> ap'}),'insert-ap'),
 frame('Create and mark app','Create the second p child, then store None at app.',state('app',{path:'root -a-> a -p-> ap -p-> app',terminal:'app=yes'}),'mark-app-terminal'),
 frame('Reuse app while inserting apple','Follow existing a, ap, app; create l->appl then e->apple and mark apple terminal.',state('apple',{operation:'insert(\"apple\")',created:'appl, apple'}),'insert-apple'),
 frame('Reject ap as a full word','search(\"ap\") reaches ap but terminal=no, so return false.',state('ap',{query:'search(\"ap\")',outcome:'false'}),'search-ap-false'),
 frame('Accept ap as a prefix','starts_with(\"ap\") only requires the node to exist, so return true.',state('ap',{query:'starts_with(\"ap\")',outcome:'true'}),'prefix-ap-true'),
 frame('Accept app as a full word','search(\"app\") reaches app with terminal=yes.',state('app',{query:'search(\"app\")',result:'true'}),'search-app-true'),
]);
export default defineVisual('implement-trie',draft,{pattern:'Character-child maps with explicit terminal markers.',recognitionCue:'Many words share prefixes and queries distinguish prefix existence from complete-word membership.',invariant:'The root-to-node edge labels spell its prefix; terminal is true exactly for inserted complete words.',stateModel:'Root dictionary, current prefix node, child maps, and terminal marker.',visualRationale:'A prefix/child/terminal grid shows sharing and exact-word semantics at readable mobile size.',rejectedAlternatives:['Independent word rows duplicate prefixes.','A flat set cannot answer prefixes.','A final tree skips character walking.'],transferLesson:'Store shared prefixes once and attach terminal or payload state to nodes.',independentReview:'3.4 source-to-frame replay',reviewStatus:'reviewed'});
