import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const ids=['dummy','A1','A4','A7','B2','B3','B8'];
const state=(next,tail,first,second,extra={})=>grid([
  ['node',...ids],['next',...ids.map(id=>next[id]??'null')],
], ids.flatMap((id,index)=>[
  ...(id===tail?[{row:0,col:index+1,label:'tail',tone:'focus',key:'tail'}]:[]),
  ...(id===first?[{row:0,col:index+1,label:'first',tone:'state',key:'first'}]:[]),
  ...(id===second?[{row:1,col:index+1,label:'second',tone:'state',key:'second'}]:[]),
]),{ input:'A:1->4->7; B:2->3->8', fixedPrefix:extra.fixedPrefix??'empty', ...extra });
const source={A1:'A4',A4:'A7',A7:'null',B2:'B3',B3:'B8',B8:'null'};

const draft=visual('Compare the two frontier nodes, link the smaller after tail, and advance only its source cursor.',[
 frame('Initialize two sorted chains','tail=dummy, first=A1, second=B2, and dummy.next=null.',state(source,'dummy','A1','B2'),'initialize-heads'),
 frame('Attach A1','1 <= 2, so dummy.next=A1; advance first to A4 and tail to A1.',state({...source,dummy:'A1'},'A1','A4','B2',{comparison:'1 <= 2',fixedPrefix:'1'}),'attach-a1'),
 frame('Attach B2','4 > 2, so A1.next=B2; advance second to B3 and tail to B2.',state({...source,dummy:'A1',A1:'B2'},'B2','A4','B3',{comparison:'4 > 2',fixedPrefix:'1->2'}),'attach-b2'),
 frame('Attach B3','4 > 3, so B2.next=B3; advance second to B8 and tail to B3.',state({...source,dummy:'A1',A1:'B2'},'B3','A4','B8',{comparison:'4 > 3',fixedPrefix:'1->2->3'}),'attach-b3'),
 frame('Attach A4','4 <= 8, so B3.next=A4; advance first to A7 and tail to A4.',state({...source,dummy:'A1',A1:'B2',B3:'A4'},'A4','A7','B8',{comparison:'4 <= 8',fixedPrefix:'1->2->3->4'}),'attach-a4'),
 frame('Attach A7','7 <= 8, so A4 already leads to A7; advance first to null and tail to A7.',state({...source,dummy:'A1',A1:'B2',B3:'A4'},'A7',null,'B8',{comparison:'7 <= 8',fixedPrefix:'1->2->3->4->7'}),'attach-a7'),
 frame('Append the remaining suffix','first is null, so set A7.next=B8 and return dummy.next.',state({...source,dummy:'A1',A1:'B2',B3:'A4',A7:'B8'},'A7',null,'B8',{fixedPrefix:'1->2->3->4->7->8',result:'[1,2,3,4,7,8]'}),'append-b8'),
]);

export default defineVisual('merge-two-sorted-lists',draft,{
 pattern:'Two sorted linked-list cursors with a dummy output head and moving tail.',recognitionCue:'Two sorted node streams must be merged by relinking existing nodes.',invariant:'The path from dummy through tail is the consumed sorted prefix; first and second begin untouched suffixes. The old link after tail may remain until the next attachment overwrites it.',stateModel:'Dummy, tail, first, second, and each node next pointer.',visualRationale:'A next-pointer grid preserves every node identity and changed link without a shrinking SVG.',rejectedAlternatives:['Copying values hides rewiring.','A final chain skips decisions.','A prose table omits pointer identity.'],transferLesson:'Emit the smaller frontier and advance only its source; append the remaining sorted suffix in constant time.',independentReview:'3.4 source-to-frame replay',reviewStatus:'reviewed',
});
