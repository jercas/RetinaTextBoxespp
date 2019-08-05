/* global web, visualization, ClassVisualization */

ClassVisualization.prototype.load_visualization = function(){

    var self = this;

    var sampleData = this.sampleData;

    var urlImg = "/image/?ch=" + getUrlParameter("ch") + "&task=" + getUrlParameter("task") + "&sample=" + getUrlParameter("sample") +  "&gtv=" + getUrlParameter("gtv");

    var template = "<div class='im_filters'><input type='checkbox' checked='checked' id='chk_image'><label for='chk_image'>Show Image</label></div>"+
                    "<div class='container_canvas'>" +
                    "<h3>Ground Truth</h3>" +
                    "<div id='div_canvas_gt'></div>" +
                   "</div>"+
                   "<div class='container_canvas'>" +
                    "<h3>Detection</h3>" +
                    "<div id='div_canvas_det'></div>" +
                   "</div>"+
                   "<img id='img_gt_image2'>"+
                   "<div id='div_sample_info'>"+
                   "<div id='div_matrices'><div class='div_table'><h3>IoU Matrix</h3>loading..</div></div>"+
                   "<div id='div_logs'><h3>Evaluation Log</h3><span class='red'>loading..</span></div>";
                    "</div>";
    
    $("#div_sample").html(template);

    if(!this.image_details_loaded){
        this.image_details_loaded=true;
        this.init_image_details();
    }   
    this.image_loaded = false;
    this.draw();
    
    $("#chk_image").change(function(){
        self.draw();
    });
    
    $("#img_gt_image2").attr("src",urlImg).one("load",function(){
        self.image_loaded = true;
        self.im_w = this.width;
        self.im_h = this.height;
        self.scale = Math.min($("#div_canvas_gt").width()/self.im_w,$("#div_canvas_det").height()/self.im_h );
        self.zoom_changed();
        self.correct_image_offset();
        self.draw();
    });

    var numGt = sampleData.gtPolPoints==undefined? 0 : sampleData.gtPolPoints.length;
    var numDet = sampleData.detPolPoints==undefined? 0 : sampleData.detPolPoints.length;

    var html_iou = "";

        var stylesMat = new Array();
        for ( var j=0;j<numGt;j++){
            stylesMat[j] = new Array();
            for ( var i=0;i<numDet;i++){
                stylesMat[j][i] = "value";
            }
        }
        
        sampleData.gtTypes = new Array();
        sampleData.detTypes = new Array();
        for ( var j=0;j<numGt;j++){
            var gtDontCare = $.inArray(j,sampleData.gtDontCare)>-1;
            sampleData.gtTypes.push( gtDontCare? 'DC' : 'NM' );
        }
        for ( var j=0;j<numDet;j++){
            var detDontCare = $.inArray(j,sampleData.detDontCare)>-1;
            sampleData.detTypes.push( detDontCare? 'DC' : 'NM' );
        }


        if (sampleData.pairs!=undefined){
            for ( var k=0;k<sampleData.pairs.length;k++){
                var pair = sampleData.pairs[k];
                
                var gts = new Array();
                var dets = new Array();
                
                if(pair.gt.length==undefined){
                    gts.push(pair.gt);
                }else{
                    gts = pair.gt;
                }
                if(pair.det.length==undefined){
                    dets.push(pair.det);
                }else{
                    dets = pair.det;
                }
                for(var i=0;i<gts.length;i++){
                    for(var j=0;j<dets.length;j++){
                        stylesMat[gts[i]][dets[j]] += " " + "OO";
                        sampleData.gtTypes[gts[i]] = "OO";
                        sampleData.detTypes[dets[j]] = "OO";
                    }
                }
            }
        }
    if(numDet>100){
        html_iou = "<p class='red'>The algorithm has detected more than 100 bounding boxes, the visualization are not posible</p></p>";
    }else{        
        
        var html_iou = "<table><thead><tr><th>GT / Det</th>";
        for ( var i=0;i<numDet;i++){
            var detDontCare = $.inArray(i,sampleData.detDontCare)>-1;
            html_iou += "<th style='" + (detDontCare? "" : "font-weight:bold;") + "'>#" + i + "</th>";
        }
        html_iou += "</tr></thead><tbody id='tbody_iou'>";

        for ( var j=0;j<numGt;j++){
            var gtDontCare = $.inArray(j,sampleData.gtDontCare)>-1;
            html_iou += "<tr>";
            html_iou += "<td style='" + (gtDontCare? "" : "font-weight:bold;") + "'>#" + j + "</td>";
            for ( var i=0;i<numDet;i++){

                var iouClass = (sampleData.iouMat[j][i]>=sampleData.evaluationParams.IOU_CONSTRAINT ? ' green' : ' red' );
                html_iou += "<td data-col='" + i + "' data-row='" + j + "' class='" + stylesMat[j][i] + " " + iouClass + "'>" + Math.round(sampleData.iouMat[j][i]*10000)/100 + "</td>";    
            }
            html_iou += "</tr>";

        }
        html_iou += "</tbody></table>";
    }

    $("#div_matrices").html("<div class='div_table'><h3>IoU Matrix</h3>" + html_iou + "</div>");
    var evalLog = sampleData.evaluationLog;
    if (evalLog==undefined){
        evalLog = "";
    }    
    $("#div_logs").html("<div class='div_log'><h3>Evaluation Log</h3>" + evalLog + "</div>");

    this.table_sizes();

    $("#div_matrices tbody td").mouseover(function(){
        self.det_rect = -1;
        self.gt_rect = -1;
        if ( $(this).attr("data-col")!=undefined && $(this).attr("data-row")!=undefined){
            self.det_rect = $(this).attr("data-col");
            self.gt_rect = $(this).attr("data-row");
            $("#div_matrices tbody td").removeClass("selected");
            $("#div_matrices tbody td").removeClass("col_selected").removeClass("row_selected");
            $(this).addClass("selected");
            $("td[data-col=" + $(this).attr("data-col") + "]").addClass("col_selected");
            $("td[data-row=" + $(this).attr("data-row") + "]").addClass("row_selected");
        }
        self.draw();
    });

    this.draw();


};

ClassVisualization.prototype.draw = function(){

    this.ctx_gt.clearRect(0,0,this.canvas_gt.width,this.canvas_gt.height);
    this.ctx_det.clearRect(0,0,this.canvas_gt.width,this.canvas_gt.height);
    
    if(!this.image_loaded){
        this.ctx_det.fillStyle = "rgba(255,0,0,1)";
        this.ctx_det.font= "12px Verdana";
        this.ctx_det.fillText("Loading image..", 20,60);
        this.ctx_gt.fillStyle = "rgba(255,0,0,1)";
        this.ctx_gt.font= "12px Verdana";
        this.ctx_gt.fillText("Loading image..", 20,60);
        
        return;
    }
    
    
    if( $("#chk_image").is(":checked")){
        this.ctx_gt.drawImage(img_gt_image2,this.offset_x,this.offset_y,this.curr_im_w,this.curr_im_h);
    }else{
        this.ctx_gt.strokeStyle = "rgba(0,0,0,1)";
        this.ctx_gt.strokeRect(this.offset_x,this.offset_y,this.curr_im_w,this.curr_im_h);
    }


    if (this.sampleData==null){
        this.ctx_gt.fillStyle = "rgba(255,0,0,1)";
        this.ctx_gt.font= "12px Verdana";
        this.ctx_gt.fillText("Loading method..", 20,60);        
        this.ctx_det.fillStyle = "rgba(255,0,0,1)";
        this.ctx_det.font= "12px Verdana";
        this.ctx_det.fillText("Loading method..", 20,60);
        return;
    }else{
         if (this.sampleData.gtPolPoints==undefined){
             this.sampleData.gtPolPoints = [];
         }
    }
    
    for (var i=0;i<this.sampleData.gtPolPoints.length;i++){
        
        //if (bb.id_s==current_id_submit){

            var opacity = 0.6;//(gt_rect==bb.i)? "0.9" : "0.6";
            
            var bb = this.sampleData.gtPolPoints[i];
            var type = this.sampleData.gtTypes[i];
            
            var gtDontCare = $.inArray(i,this.sampleData.gtDontCare)>-1;
            
            if(type=="DC"){
                this.ctx_gt.fillStyle = "rgba(50,50,50," + opacity + ")";
            }else if (type=="OO"){
                this.ctx_gt.fillStyle = "rgba(0,190,0," + opacity + ")";
            }else if (type=="NO"){
                this.ctx_gt.fillStyle = "rgba(38,148,232," + opacity + ")";                
            }else{
                this.ctx_gt.fillStyle = "rgba(255,0,0," + opacity + ")";
            }

            if (bb.length==4){

                var x = this.original_to_zoom_val(parseInt(bb[0]));
                var y = this.original_to_zoom_val_y(parseInt(bb[1]));
                var x2 = this.original_to_zoom_val(parseInt(bb[2]));
                var y2 = this.original_to_zoom_val_y(parseInt(bb[3]));
                var w = x2-x+1;
                var h = y2-y+1;
                this.ctx_gt.fillRect(x,y,w,h);
                if(this.gt_rect==i){
                    this.ctx_gt.lineWidth = 2;
                    this.ctx_gt.strokeStyle = 'red';
                    this.ctx_gt.strokeRect(x,y,w,h);
                }  
            }else{
                this.ctx_gt.beginPath();
                this.ctx_gt.moveTo(this.original_to_zoom_val(parseInt(bb[0])), this.original_to_zoom_val_y(parseInt(bb[1])));
                this.ctx_gt.lineTo(this.original_to_zoom_val(parseInt(bb[2])+1), this.original_to_zoom_val_y(parseInt(bb[3])));
                this.ctx_gt.lineTo(this.original_to_zoom_val(parseInt(bb[4])+1), this.original_to_zoom_val_y(parseInt(bb[5])+1));
                this.ctx_gt.lineTo(this.original_to_zoom_val(parseInt(bb[6])), this.original_to_zoom_val_y(parseInt(bb[7])+1));
                this.ctx_gt.closePath();
                this.ctx_gt.fill();

                //ctx_gt.fillRect( original_to_zoom_val(parseInt(bb.x)),original_to_zoom_val_y(parseInt(bb.y)),parseInt(bb.w)*scale,parseInt(bb.h)*scale);
                if(this.gt_rect==i){
                    this.ctx_gt.lineWidth = 2;
                    this.ctx_gt.strokeStyle = 'red';
                    this.ctx_gt.stroke();
                }            
            }

        //}
    }


    this.ctx_det.clearRect(0,0,this.canvas_gt.width,this.canvas_gt.height);
    if( $("#chk_image").is(":checked")){
        this.ctx_det.drawImage(img_gt_image2,this.offset_x,this.offset_y,this.curr_im_w,this.curr_im_h);
    }else{
        this.ctx_det.strokeStyle = "rgba(0,0,0,1)";
        this.ctx_det.strokeRect(this.offset_x,this.offset_y,this.curr_im_w,this.curr_im_h);
    }    

    for (var i=0;i<this.sampleData.detPolPoints.length;i++){
        var bb = this.sampleData.detPolPoints[i];
        var type = this.sampleData.detTypes[i];

            var opacity = 0.6;//(det_rect==bb.i)? "0.9" : "0.6";
            if(type=="DC"){
                this.ctx_det.fillStyle = "rgba(50,50,50," + opacity + ")";
            }else if (type=="OO"){
                this.ctx_det.fillStyle = "rgba(0,190,0," + opacity + ")";
            }else if (type=="NO"){
                this.ctx_det.fillStyle = "rgba(38,148,232," + opacity + ")";                
            }else{
                this.ctx_det.fillStyle = "rgba(255,0,0," + opacity + ")";
            }

            if (bb.length==4){
                
                var x = this.original_to_zoom_val(parseInt(bb[0]));
                var y = this.original_to_zoom_val_y(parseInt(bb[1]));
                var x2 = this.original_to_zoom_val(parseInt(bb[2]));
                var y2 = this.original_to_zoom_val_y(parseInt(bb[3]));
                var w = x2-x+1;
                var h = y2-y+1;
                this.ctx_det.fillRect(x,y,w,h);
                if(this.det_rect==i){
                    this.ctx_det.lineWidth = 2;
                    this.ctx_det.strokeStyle = 'red';
                    this.ctx_det.strokeRect(x,y,w,h);
                }   
                
            }else{
                this.ctx_det.beginPath();
                this.ctx_det.moveTo(this.original_to_zoom_val(parseInt(bb[0])), this.original_to_zoom_val_y(parseInt(bb[1])));
                this.ctx_det.lineTo(this.original_to_zoom_val(parseInt(bb[2])+1), this.original_to_zoom_val_y(parseInt(bb[3])));
                this.ctx_det.lineTo(this.original_to_zoom_val(parseInt(bb[4])+1), this.original_to_zoom_val_y(parseInt(bb[5])+1));
                this.ctx_det.lineTo(this.original_to_zoom_val(parseInt(bb[6])), this.original_to_zoom_val_y(parseInt(bb[7])+1));
                this.ctx_det.closePath();
                this.ctx_det.fill();

                //ctx_gt.fillRect( original_to_zoom_val(parseInt(bb.x)),original_to_zoom_val_y(parseInt(bb.y)),parseInt(bb.w)*scale,parseInt(bb.h)*scale);
                if(this.det_rect==i){
                    this.ctx_det.lineWidth = 2;
                    this.ctx_det.strokeStyle = 'red';
                    this.ctx_det.stroke();
                }   
            }
    }
    this.draws++;
};